import sympy as sp
from ..meta import (
    DType,
    funcname,
    watermarked,
    assert_name,
    get_parameters,
)
from ..dense import GenDense
import itertools


def nalgebra_template(name, params, dtype, return_shape):
    assert_name(name)
    assert (
        len(return_shape) == 2
    ), "Return shape shoule have 2 dimensions, found {return_shape}"

    m, n = return_shape
    range_prod = itertools.product(range(m), range(n))
    param_list = ", ".join([f"{p}: {str(dtype)}" for p in params])
    param_invoke = ", ".join(params)

    def entry_assign(mi, ni):
        return f"""
result[({mi}, {ni})] = {funcname(name, mi, ni)}({param_invoke});
"""

    assigns = "\n".join([entry_assign(mi, ni) for mi, ni in range_prod])
    return f"""
pub fn {name}({param_list}) -> nalgebra::SMatrix<{str(dtype)}, {m}, {n}> {{
    let mut result = nalgebra::SMatrix::zeros();

    {assigns}

    result
}}
"""


class GenNalgebra:
    def __init__(
        self,
        dtype: DType,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
        self.dtype = dtype
        self.dense = GenDense(dtype, tol, precision_digit, debug)

    def generate(self, func_name: str, mat: sp.Matrix):
        entries_impl = self.dense.generate(func_name, mat)
        params = get_parameters(mat)
        entries_impl["matrix"] = nalgebra_template(
            func_name, params, self.dtype, mat.shape
        )

        output_code = "\n".join(entries_impl.values())

        return watermarked(output_code)
