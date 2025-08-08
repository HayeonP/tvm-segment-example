#!/bin/bash

g++ -std=c++17 \
    -o 0_get_runtime_sequence 0_get_runtime_sequence.cpp \
    -I/home/rubis/workspace/tvm-segment-21/include \
    -I/home/rubis/workspace/tvm-segment-21/ffi/include \
    -I/home/rubis/workspace/tvm-segment-21/3rdparty/dmlc-core/include \
    -I/home/rubis/workspace/tvm-segment-21/ffi/3rdparty/dlpack/include \
    -I/usr/include/opencv4 \
    -L/home/rubis/workspace/tvm-segment-21/build \
    -lpthread -lcurl -ljpeg -ldeflate \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
    -ltvm_runtime


g++ -std=c++17 \
    -o 1_segment_runner 1_segment_runner.cpp \
    -I/home/rubis/workspace/tvm-segment-21/include \
    -I/home/rubis/workspace/tvm-segment-21/ffi/include \
    -I/home/rubis/workspace/tvm-segment-21/3rdparty/dmlc-core/include \
    -I/home/rubis/workspace/tvm-segment-21/ffi/3rdparty/dlpack/include \
    -I/usr/include/opencv4 \
    -L/home/rubis/workspace/tvm-segment-21/build \
    -lpthread -lcurl -ljpeg -ldeflate \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
    -ltvm_runtime
