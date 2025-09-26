import tvm
import numpy as np

# --------------------------
# 4) 실행 테스트
# --------------------------

dev = tvm.device("cuda", 0)
ex = tvm.runtime.load_module('factorial_cuda.so')
vm = tvm.relax.VirtualMachine(ex, dev)

n = 3
gpu_input = tvm.nd.array(np.array([n], dtype="int32"), dev)
param_arr = tvm.nd.array(np.array([4], dtype="int32"), dev)  # param=4
ret = vm["main"](gpu_input, param_arr)

print(f"factorial({n}) * param = {ret.numpy()[0]}")