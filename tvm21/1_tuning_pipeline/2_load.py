import os
import tvm
from tvm import relax
from tvm.runtime import load_module
import numpy as np


def load_params(path):                
    param_dict = tvm.runtime.load_param_dict_from_file(path)
    ordered_keys = sorted(param_dict.keys(), key=lambda k: int(k.split("_")[1]))
    return {"main": [param_dict[k] for k in ordered_keys]}    


if __name__ == "__main__":
    dev = tvm.device("cuda", 0)
    ex = tvm.runtime.load_module('resnet18.so')
    
    params = load_params("resnet18.bin")

    vm = relax.VirtualMachine(ex, dev)
    
    input_data = np.random.rand(1,3,224,224).astype("float32")
    gpu_input = tvm.nd.array(input_data, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]
    output = vm['main'](gpu_input, *gpu_params)
    
    print(output)