#include <dlpack/dlpack.h>
#include <tvm/runtime/segment_runner.h>
#include <tvm/runtime/cpp_utils.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/device_api.h>

#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

std::string readFileAsString(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file) {
        throw std::runtime_error("파일을 열 수 없습니다: " + filePath);
    }

    std::string content(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    file.close();
    return content;
}

cv::Mat loadImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    return image;
}

std::vector<float> preprocessImage(
    const cv::Mat& image, 
    int resize_size = 256, 
    int crop_size = 224,
    const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
    const std::vector<float>& std = {0.229f, 0.224f, 0.225f}
) {
    cv::Mat processed = image.clone();

    // 1. 리사이즈
    double aspect_ratio = static_cast<double>(processed.cols) / processed.rows;
    int new_width, new_height;
    if (processed.cols > processed.rows) {
        new_height = resize_size;
        new_width = static_cast<int>(resize_size * aspect_ratio);
    } else {
        new_width = resize_size;
        new_height = static_cast<int>(resize_size / aspect_ratio);
    }
    cv::resize(processed, processed, cv::Size(new_width, new_height));

    // 2. 크롭
    int x = (processed.cols - crop_size) / 2;
    int y = (processed.rows - crop_size) / 2;
    if (x < 0 || y < 0) {
        throw std::runtime_error("Crop size larger than resized image");
    }
    cv::Rect roi(x, y, crop_size, crop_size);
    processed = processed(roi).clone();

    // 3. float 변환 및 정규화, HWC → CHW 변환
    const int channels = 3;
    const int height = processed.rows;
    const int width = processed.cols;
    if (processed.channels() != channels) {
        throw std::runtime_error("Input image must have 3 channels");
    }

    cv::Mat float_image;
    processed.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    std::vector<float> tensor_data(1 * channels * height * width);
    const int hw_size = height * width;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float value = float_image.at<cv::Vec3f>(h, w)[c];
                tensor_data[c * hw_size + h * width + w] = (value - mean[c]) / std[c];
            }
        }
    }
    return tensor_data;
}

std::vector<std::string> loadLabels(const std::string& file_path) {
    std::vector<std::string> labels;
    std::ifstream file(file_path);
    std::string line;
    
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

std::string getLabel(const tvm::runtime::NDArray& output, const std::vector<std::string>& labels) {
    // Extract data
    const DLTensor* tensor = output.operator->();
    const float* data = static_cast<float*>(tensor->data);
    int64_t size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        size *= tensor->shape[i];
    }

    // argmax
    int max_index = 0;
    float max_value = data[0];
    for (int64_t i = 1; i < size; ++i) {
        if (data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }
    
    if (max_index >= 0 && max_index < labels.size()) {
        return labels[max_index];
    }

    return "Unknown";
}

float calculate_duration_avg(std::vector<std::chrono::duration<float>> list){
    if (list.empty()) return 0.0f;

    float sum = 0.0f;
    for (const auto& dur : list) {
        sum += dur.count();
    }
    return sum / list.size();
}

float calculate_duration_max(std::vector<std::chrono::duration<float>> list){
    if (list.empty()) return 0.0f;

    auto max_itr = std::max_element(list.begin(), list.end(),
        [](const auto& a, const auto& b) { return a.count() < b.count(); });
    return max_itr->count();
}


int main() {
    std::vector<std::string> image_path_list;
    image_path_list.push_back("../data/dog.jpg");
    image_path_list.push_back("../data/Gatto_europeo4.jpg");
    image_path_list.push_back("../data/960px-African_Bush_Elephant.jpg");

    // dev = tvm.device("cuda", 0)
    tvm::Device dev{kDLCUDA, 0}; // OK
    // get device api for synchronization
    tvm::runtime::DeviceAPI* device_api = tvm::runtime::DeviceAPI::Get(dev);
    
    // ex = tvm.runtime.load_module("resnet18.so")   
    std::string library_apth{"resnet18.so"};
    
    auto exec = tvm::runtime::LoadExecutableModule(library_apth);

    // params = load_params("resnet18.bin")    
    std::string binary_path{"resnet18.bin"};
    std::vector<tvm::runtime::NDArray> params = tvm::runtime::LoadParamsAsNDArrayList(binary_path); 

    tvm::runtime::SegmentRunner segment_runner(exec, dev);
    
    // segments_length = segment_runner.load(segments_info)
    std::string runtime_sequence = readFileAsString("runtime_sequence");
    segment_runner.Load(runtime_sequence);
    size_t segments_length = segment_runner.GetLength();

    // gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]
    std::vector<tvm::runtime::NDArray> gpu_params;
    for(auto& param : params){
        gpu_params.push_back(param.CopyTo(dev));
    }

    //  labels = get_imagenet_labels()
    std::vector<std::string> labels = loadLabels("../data/imagenet_classes.txt");

    int image_idx = 0;
    std::vector<std::chrono::duration<float>> gpu_input_time_list;
    std::vector<std::chrono::duration<float>> set_input_with_params_time_list;
    std::vector<std::chrono::duration<float>> inference_time_list;
    std::vector<std::chrono::duration<float>> gpu_output_time_list;
    std::vector<std::chrono::duration<float>> output_time_list;
    std::vector<std::chrono::duration<float>> get_label_time_list;
    std::vector<std::chrono::duration<float>> total_time_list;

    for(int i = 0; i < 120; i++){
        std::string image_path = image_path_list[image_idx];
        try {
            // Load input
            cv::Mat image = loadImage(image_path);
            std::vector<float> preprocessed_image = preprocessImage(image, 256, 224);
            std::vector<int64_t> shape = {1, 3, 224, 224}; // NCHW
            
            auto total_start = std::chrono::high_resolution_clock::now();


            // gpu input - Copy input to GPU
            auto gpu_input_start = std::chrono::high_resolution_clock::now();
            tvm::runtime::NDArray input = tvm::runtime::convertVecToNDArray(preprocessed_image, shape);
            tvm::runtime::NDArray gpu_input = input.CopyTo(dev);
            auto gpu_input_end = std::chrono::high_resolution_clock::now();
            gpu_input_time_list.push_back(gpu_input_end - gpu_input_start);
            

            // set input with params - Set input to segment runner
            auto set_input_with_params_start = std::chrono::high_resolution_clock::now();
            std::vector<tvm::runtime::NDArray> input_vec;
            input_vec.push_back(gpu_input);
            segment_runner.SetInputWithParams(input_vec, gpu_params);
            auto set_input_with_params_end = std::chrono::high_resolution_clock::now();
            set_input_with_params_time_list.push_back(set_input_with_params_end - set_input_with_params_start);


            // inference - Run segments            
            auto inference_start = std::chrono::high_resolution_clock::now();
            TVMStreamHandle stream = device_api->GetCurrentStream(dev); // Get the current default stream for the device
            for(size_t i = 0; i < segments_length; i++){                
                segment_runner.Execute(i);
                device_api->StreamSync(dev, stream); // force synchronization
            }
            auto inference_end = std::chrono::high_resolution_clock::now();
            inference_time_list.push_back(inference_end - inference_start);

            
            // gpu output - Get output
            auto gpu_output_start = std::chrono::high_resolution_clock::now();
            std::vector<tvm::runtime::NDArray> gpu_output = segment_runner.GetOutput();
            auto gpu_output_end = std::chrono::high_resolution_clock::now();
            gpu_output_time_list.push_back(gpu_output_end - gpu_output_start);

            // Copy output to Host
            auto output_start = std::chrono::high_resolution_clock::now();
            tvm::runtime::NDArray output = gpu_output[0].CopyTo(DLDevice{kDLCPU, 0});
            auto output_end = std::chrono::high_resolution_clock::now();
            output_time_list.push_back(output_end - output_start);

            // get label - Find label
            auto get_label_start = std::chrono::high_resolution_clock::now();
            std::cout << getLabel(output, labels) << std::endl;
            auto get_label_end = std::chrono::high_resolution_clock::now();
            get_label_time_list.push_back(get_label_end - get_label_start);

            auto total_end = std::chrono::high_resolution_clock::now();
            total_time_list.push_back(total_end - total_start);

            image_idx = (image_idx + 1) % image_path_list.size();
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    } 

    gpu_input_time_list.erase(gpu_input_time_list.begin(), gpu_input_time_list.begin()+20);
    set_input_with_params_time_list.erase(set_input_with_params_time_list.begin(), set_input_with_params_time_list.begin()+20);
    inference_time_list.erase(inference_time_list.begin(), inference_time_list.begin()+20);
    gpu_output_time_list.erase(gpu_output_time_list.begin(), gpu_output_time_list.begin()+20);
    output_time_list.erase(output_time_list.begin(), output_time_list.begin()+20);
    get_label_time_list.erase(get_label_time_list.begin(), get_label_time_list.begin()+20);
    total_time_list.erase(total_time_list.begin(), total_time_list.begin()+20);
                  
    float gpu_input_time_avg = calculate_duration_avg(gpu_input_time_list) * 1000;
    float gpu_input_time_max = calculate_duration_max(gpu_input_time_list) * 1000;
    float set_input_with_params_time_avg = calculate_duration_avg(set_input_with_params_time_list) * 1000;
    float set_input_with_params_time_max = calculate_duration_max(set_input_with_params_time_list) * 1000;
    float inference_time_avg = calculate_duration_avg(inference_time_list) * 1000;
    float inference_time_max = calculate_duration_max(inference_time_list) * 1000;
    float gpu_output_time_avg = calculate_duration_avg(gpu_output_time_list) * 1000;
    float gpu_output_time_max = calculate_duration_max(gpu_output_time_list) * 1000;
    float output_time_avg = calculate_duration_avg(output_time_list) * 1000;
    float output_time_max = calculate_duration_max(output_time_list) * 1000;
    float get_label_time_avg = calculate_duration_avg(get_label_time_list) * 1000;
    float get_label_time_max = calculate_duration_max(get_label_time_list) * 1000;
    float total_time_avg = calculate_duration_avg(total_time_list) * 1000;
    float total_time_max = calculate_duration_max(total_time_list) * 1000;

    std::cout << "===========================" << std::endl;
    std::cout << "# gpu input time" << std::endl;
    std::cout << "Average response time: " << gpu_input_time_avg << "ms" << std::endl;
    std::cout << "Worst response time: " << gpu_input_time_max << "ms" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "# set input with params time" << std::endl;
    std::cout << "Average response time: " << set_input_with_params_time_avg << "ms" << std::endl;
    std::cout << "Worst response time: " << set_input_with_params_time_max << "ms" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "# inference time" << std::endl;
    std::cout << "Average response time: " << inference_time_avg << "ms" << std::endl;
    std::cout << "Worst response time: " << inference_time_max << "ms" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "# gpu output time" << std::endl;
    std::cout << "Average response time: " << gpu_output_time_avg << "ms" << std::endl;
    std::cout << "Worst response time: " << gpu_output_time_max << "ms" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "# output time" << std::endl;
    std::cout << "Average response time: " << output_time_avg << "ms" << std::endl;
    std::cout << "Worst response time: " << output_time_max << "ms" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "# get label time" << std::endl;
    std::cout << "Average response time: " << get_label_time_avg << "ms" << std::endl;
    std::cout << "Worst response time: " << get_label_time_max << "ms" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "# total time" << std::endl;
    std::cout << "Average response time: " << total_time_avg << "ms" << std::endl;
    std::cout << "Worst response time: " << total_time_max << "ms" << std::endl;


    return 0;
}

