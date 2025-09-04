import os
import tvm
from tvm import relax
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch
import time
from tqdm import tqdm

def load_params(path):                
    param_dict = tvm.runtime.load_param_dict_from_file(path)
    ordered_keys = sorted(param_dict.keys(), key=lambda k: int(k.split("_")[1]))
    return {"main": [param_dict[k] for k in ordered_keys]}


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).numpy()  # (1, 3, 224, 224)
    return image, img_tensor


def get_imagenet_labels():
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    file_path = "imagenet_classes.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    with open(file_path) as f:
        return [line.strip() for line in f.readlines()]


def save_prediction(image, label, path, title="Prediction"):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.title(f"{title}: {label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)


def infer_with_tvm(image_tensor: np.ndarray, params_path: str, lib_path: str, device="cuda"):
    dev = tvm.device(device, 0)
    ex = tvm.runtime.load_module(lib_path)
    params = load_params(params_path)
    vm = relax.VirtualMachine(ex, dev)

    gpu_input = tvm.nd.array(image_tensor.astype("float32"), dev)
    gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]

    _ = vm["main"](gpu_input, *gpu_params)  # warm-up
    dev.sync()

    start = time.time()
    output = vm["main"](gpu_input, *gpu_params)
    dev.sync()
    end = time.time()

    print('tvm:', end - start)
    output_nd = output[0] if isinstance(output, tvm.ir.Array) else output
    return np.argmax(output_nd.numpy())


def infer_with_torch(image_tensor: np.ndarray):
    image_tensor = torch.from_numpy(image_tensor).to(torch.float32).cuda()
    model = resnet18(weights=ResNet18_Weights.DEFAULT).cuda().eval()
    
    with torch.no_grad():
        _ = model(image_tensor)  # warm-up
        torch.cuda.synchronize()

        start = time.time()
        output = model(image_tensor)
        torch.cuda.synchronize()
        end = time.time()

    print('pytorch:', end - start)
    return int(torch.argmax(output, dim=1).item())

def get_imagenet_labels():
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    file_path = "imagenet_classes.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    with open(file_path) as f:
        return [line.strip() for line in f.readlines()]

def get_label(output):
    output_nd = output[0] if isinstance(output, tvm.ir.Array) else output
    prediction = np.argmax(output_nd.numpy())
    labels = get_imagenet_labels()
    return labels[prediction]

if __name__ == "__main__":
    # 샘플 이미지 로드    
    image_sources = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/1/14/Gatto_europeo4.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/960px-African_Bush_Elephant.jpg",        
    ]

    image_path_list = []
    for source in image_sources:
        image_path = source.split('/')[-1]
        if not os.path.exists(image_path):
            import urllib.request
            urllib.request.urlretrieve(
                source, image_path)
        image_path_list.append(image_path)
    
    dev = tvm.device("cuda", 0)
    ex = tvm.runtime.load_module("resnet18.so")
    params = load_params("resnet18.bin")
    vm = relax.VirtualMachine(ex, dev)
    
    gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]
    image_idx = 0
    
    gpu_input_time_list = []
    inference_time_list = []
    get_label_time_list = []
    total_time_list = []
    
    for _ in tqdm(range(120)):
        image_path = image_path_list[image_idx]
        
        
        orig_image, image_tensor = preprocess_image(image_path)
        labels = get_imagenet_labels()

        total_start = time.time()

        # GPU input
        start = time.time()
        gpu_input = tvm.nd.array(image_tensor.astype("float32"), dev)
        end = time.time()
        gpu_input_time_list.append(end-start)
        
        # Default inference
        start = time.time()
        output = vm["main"](gpu_input, *gpu_params)
        end = time.time()
        inference_time_list.append(end-start)
        
        # get label
        start = time.time()
        get_label(output)
        end = time.time()
        get_label_time_list.append(end-start)

        total_end = end
        total_time_list.append(total_end - total_start)
        
        image_idx = (image_idx + 1) % len(image_path_list)
    
    gpu_input_time_list = gpu_input_time_list[21:]
    inference_time_list = inference_time_list[21:]
    total_time_list = total_time_list[21:]
    
    gpu_input_time_avg = sum(gpu_input_time_list)/len(gpu_input_time_list)*1000
    gpu_input_time_max = max(gpu_input_time_list)*1000
    inference_time_avg = sum(inference_time_list)/len(inference_time_list)*1000
    inference_time_max = max(inference_time_list)*1000
    get_label_time_avg = sum(get_label_time_list)/len(get_label_time_list)*1000
    get_label_time_max = max(get_label_time_list)*1000
    total_time_avg = sum(total_time_list)/len(total_time_list)*1000
    total_time_max = max(total_time_list)*1000
    
    print("===========================")
    print("# gpu intput time")
    print(f"Average response time: {gpu_input_time_avg}ms")
    print(f"Worst response time: {gpu_input_time_max}ms")
    print("===========================")
    print("# inference time")
    print(f"Average response time: {inference_time_avg}ms")
    print(f"Worst response time: {inference_time_max}ms")
    print("===========================")
    print("# get label time")
    print(f"Average response time: {get_label_time_avg}ms")
    print(f"Worst response time: {get_label_time_max}ms")
    print("===========================")
    print("# total time")
    print(f"Average response time: {total_time_avg}ms")
    print(f"Worst response time: {total_time_max}ms")
    