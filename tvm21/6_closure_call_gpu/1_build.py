from tvm.script import tir as T
from tvm.script import relax as R
import tvm
import numpy as np

@tvm.script.ir_module
class FactorialModuleCUDA:
    @T.prim_func
    def factorial_impl(n: T.Buffer((1,), "int32"),
                       result: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "factorial_impl", "tir.noalias": True})
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 1)
        result[0] = T.int32(1)
        for i in T.serial(n[0]):
            result[0] = result[0] * (i + T.int32(1))

    # --- GPU add kernel (1-element) ---
    @T.prim_func
    def add_impl(a: T.Buffer((1,), "int32"),
                 b: T.Buffer((1,), "int32"),
                 out: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "add_impl", "tir.noalias": True})
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 1)
        out[0] = a[0] + b[0]

    # --- GPU mul kernel (1-element) ---
    @T.prim_func
    def mul_impl(a: T.Buffer((1,), "int32"),
                 b: T.Buffer((1,), "int32"),
                 out: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "mul_impl", "tir.noalias": True})
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 1)
        out[0] = a[0] * b[0]

    @R.function
    def bar(n: R.Tensor((1,), "int32")) -> R.Tensor((1,), "int32"):
        return R.call_tir(
            FactorialModuleCUDA.factorial_impl,
            (n,),
            out_sinfo=R.Tensor((1,), "int32"),
        )

    @R.function
    def foo(n: R.Tensor((1,), "int32"),
            param: R.Tensor((1,), "int32")) -> R.Tensor((1,), "int32"):
        fact = FactorialModuleCUDA.bar(n)
        # 대신 fact + param -> call_tir(add_impl)
        return R.call_tir(
            FactorialModuleCUDA.add_impl,
            (fact, param),
            out_sinfo=R.Tensor((1,), "int32"),
        )

    @R.function
    def main(n: R.Tensor((1,), "int32"),
             param: R.Tensor((1,), "int32")) -> R.Tensor((1,), "int32"):
        tmp = FactorialModuleCUDA.foo(n, param)
        # 대신 tmp * param -> call_tir(mul_impl)
        return R.call_tir(
            FactorialModuleCUDA.mul_impl,
            (tmp, param),
            out_sinfo=R.Tensor((1,), "int32"),
        )



target = tvm.target.Target("cuda")      # ✅ CUDA 타깃
ex = tvm.compile(FactorialModuleCUDA, target=target)
ex.export_library("factorial_cuda.so")