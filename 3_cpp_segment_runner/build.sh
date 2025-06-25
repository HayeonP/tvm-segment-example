g++ -std=c++17 \
    -o main main.cpp \
    -I/home/hayeon/workspace/tvm/tvm20/tvm-segment/include \
    -I/home/hayeon/workspace/tvm/tvm20/tvm-segment/3rdparty/dmlc-core/include \
    -I/home/hayeon/workspace/tvm/tvm20/tvm-segment/3rdparty/dlpack/include \
    -L/home/hayeon/workspace/tvm/tvm20/tvm-segment/build \
    -ltvm_allvisible \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs