import tvm
import tvm.relax as relax
import onnx
import os
from tvm.relax.frontend import onnx as tvm_onnx

"""
SETP 1: Convert the ONNX model
"""
onnx_mod = onnx.load("data/yolo11n.onnx")

input_name = "input1"
shape_dict = {input_name: [1, 3, 224, 224]}

ir_mod = tvm_onnx.from_onnx(onnx_mod, shape_dict=shape_dict)

mod, params = relax.frontend.detach_params(ir_mod)

"""
STEP 2: TUNING
"""
IS_IN_CI = os.getenv("CI", "") == "true"

TOTAL_TRIALS = 90*8
target = tvm.target.Target("nvidia/jetson-agx-orin-64gb")
work_dir = "tuning_logs"

if not IS_IN_CI:
    mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=TOTAL_TRIALS)(mod)



"""
STEP 3: SAVE
"""
def save_params(params, path):
    if not isinstance(params, dict):
        raise ValueError("The params is not the dict")    
    
    param_dict = {f"param_{i}": p for i, p in enumerate(params["main"])}
    tvm.runtime.save_param_dict_to_file(param_dict, path)
    
    return

if not IS_IN_CI:
    ex = tvm.compile(mod, target=target)
    ex.export_library('yolo11n.so')
    
    from tvm.ir import save_json
    with open('yolo11n_relax.json', 'w') as f:
        f.write(save_json(mod))
    
    save_params(params, 'yolo11n.bin')