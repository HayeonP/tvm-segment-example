#!/bin/bash

g++ -std=c++20 \
    -o 3_cpp_segment_runner 3_cpp_segment_runner.cpp \
    -I/home/rubis/workspace/tvm-segment-21/include \
    -I/home/rubis/workspace/tvm-segment-21/ffi/include \
    -I/home/rubis/workspace/tvm-segment-21/3rdparty/dmlc-core/include \
    -I/home/rubis/workspace/tvm-segment-21/ffi/3rdparty/dlpack/include \
    -I/usr/include/opencv4 \
    -L/home/rubis/workspace/tvm-segment-21/build \
    -lpthread -lcurl -ljpeg -ldeflate \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
    -ltvm_runtime