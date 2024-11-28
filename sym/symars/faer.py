import sympy as sp
from .meta import DType, funcname, watermarked, assert_name
from .matrix_base import SymarsDense
import itertools


def faer_template(name, params, dtype_str, return_shape):
    assert_name(name)
    assert (
        len(return_shape) == 2
    ), "Return shape shoule have 2 dimensions, found {return_shape}"

    m, n = return_shape
    range_prod = itertools.product(range(m), range(n))
    param_list = ", ".join([f"{p}: {dtype_str}" for p in params])
    param_invoke = ", ".join(params)

    matmut_type = f"faer::MatMut<{dtype_str}>"

    def entry_assign(mi, ni):
        return f"""
mat[({mi}, {ni})] = {funcname(name, mi, ni)}({param_invoke});
"""

    assigns = "\n".join([entry_assign(mi, ni) for mi, ni in range_prod])
    return f"""
pub fn {name}(mut mat: {matmut_type}, {param_list}) {{

    {assigns}
}}
"""


class SymarsFaer:
    def __init__(self, dtype: DType, tol: float = 1e-9, debug: bool = False):
        self.dtype = dtype
        self.dense = SymarsDense(dtype, tol, debug)

    def generate(self, mat: sp.Matrix, func_name: str):
        entries_impl = self.dense.generate(mat, func_name)
        params = self.dense.params(mat)
        entries_impl["matrix"] = faer_template(
            func_name, params, str(self.dtype), mat.shape
        )

        output_code = "\n".join(entries_impl.values())

        return watermarked(output_code)
