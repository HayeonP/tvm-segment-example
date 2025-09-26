import tvm
import numpy as np
from tvm.runtime.segment_runner import SegmentRunner

# --------------------------
# 4) 실행 테스트
# --------------------------

dev = tvm.cpu()
ex = tvm.runtime.load_module('factorial.so')
segment_runner = SegmentRunner(ex, dev)
runtime_sequence = segment_runner.get_runtime_sequence()
with open("runtime_sequence", 'w', encoding='utf-8') as f:
    f.write(runtime_sequence)

# vm = tvm.relax.VirtualMachine(ex, dev)

# n = 3
# param_arr = tvm.nd.array(np.array([4], dtype="int32"), dev)  # param=4
# ret = vm["main"](n, param_arr)

# print(f"factorial({n}) * param = {ret.numpy()[0]}")