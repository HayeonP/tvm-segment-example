import tvm
import numpy as np
from tvm.runtime.segment_runner import SegmentRunner


# --------------------------
# 4) 실행 테스트
# --------------------------

# dev = tvm.device("cuda", 0)
dev = tvm.cpu()
ex = tvm.runtime.load_module('factorial_cuda.so')
vm = tvm.relax.VirtualMachine(ex, dev)

segment_runner = SegmentRunner(ex, dev)

params = tvm.nd.array(np.array([4], dtype="int32"), dev)  # param=4

with open("runtime_sequence", "r") as f:
    runtime_sequence = f.read()    
segment_runner.load(runtime_sequence)

segments_length = len(segment_runner.segment_list)

# Set params at the first
gpu_params = [params]
n= 3
gpu_input = tvm.nd.array(np.array([n], dtype="int32"), dev)
segment_runner.set_input_with_params(gpu_input, gpu_params)
for i in range(segments_length):
    print("Run segment:", i)
    segment_runner.execute(i)

gpu_output = segment_runner.get_output()
output = gpu_output.copyto(tvm.cpu(0))

print(f"factorial({n}) * param = {output}")