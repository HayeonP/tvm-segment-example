import tvm
from tvm.script import tir as T
from tvm.script import relax as R
import numpy as np

# --------------------------
# 1) IRModule 정의 (TIR + Relax)
# --------------------------
@tvm.script.ir_module
class FactorialModule:
    # TIR: n과 result를 Buffer로 선언
    @T.prim_func
    def factorial_impl(n: T.Buffer((1,), "int32"),
                       result: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "factorial_impl", "tir.noalias": True})
        result[0] = T.int32(1)
        for i in T.serial(n[0]):      # 권장: T.serial(동적 extent)
            result[0] = result[0] * (i + T.int32(1))

    # Relax: bar → factorial_impl 호출
    @R.function
    def bar(n: R.Tensor((1,), "int32")) -> R.Tensor((1,), "int32"):
        return R.call_tir(
            FactorialModule.factorial_impl,
            (n,),                                  # TIR의 첫 인자 n(Buffer)
            out_sinfo=R.Tensor((1,), "int32"),     # result(Buffer)를 생성해 연결
        )

    # Relax: foo → bar 호출 + param 더하기
    @R.function
    def foo(n: R.Tensor((1,), "int32"),
            param: R.Tensor((1,), "int32")) -> R.Tensor((1,), "int32"):
        fact = FactorialModule.bar(n)
        return fact + param

    # Relax: main → foo 호출 + param 곱하기
    @R.function
    def main(n: R.Tensor((1,), "int32"),
             param: R.Tensor((1,), "int32")) -> R.Tensor((1,), "int32"):
        tmp = FactorialModule.foo(n, param)
        return tmp * param               # 여기서 Relax 연산(multiply)


# --------------------------
# 2) 파라미터 저장 (param_1 = 2)
# --------------------------
def save_params(path):
    param_dict = {"param_1": tvm.nd.array(np.array([2], dtype="int32"))}
    param_bytes = tvm.runtime.save_param_dict(param_dict)
    with open(path, "wb") as f:
        f.write(param_bytes)

params = {"param_1": tvm.nd.array(np.array([2], dtype="int32"))}

# --------------------------
# 3) compile + export
# --------------------------
target = tvm.target.Target("llvm")
ex = tvm.compile(FactorialModule, target=target)
ex.export_library("factorial.so")

from tvm.ir import save_json
with open("factorial_relax.json", "w") as f:
    f.write(save_json(FactorialModule))

param_bytes = tvm.runtime.save_param_dict(params)
with open("factorial.bin", "wb") as f:
    f.write(param_bytes)

print("✅ tvm.compile + .so/.json/.bin 저장 완료")
