from ..meta import (
    DType,
    watermarked,
    assert_name,
    get_parameters,
)
from ..dense import GenDense
from .vecshape import VecShape, get_vecshape
import sympy as sp


def array_vector_template(name, params, vecshape: VecShape, dtype: DType, return_shape):
    assert_name(name)
    assert (
        len(return_shape) == 2
    ), "Return shape shoule have 2 dimensions, found {return_shape}"

    m, n = return_shape
    numel = m * n
    param_list = ", ".join([f"{p}: {str(dtype)}" for p in params])
    param_invoke = ", ".join(params)

    arr_type = f"[{str(dtype)}; {numel}]"

    def entry_assign(i):
        return f"""
vec[{i}] = {vecshape.func_name(name, i)}({param_invoke});
"""

    assigns = "\n".join([entry_assign(i) for i in range(numel)])
    return f"""
pub fn {name}({param_list}) -> {arr_type} {{

    let mut vec: {arr_type} = [0{dtype.suffix()}; {numel}];
    {assigns}
    
    vec
}}
"""


class GenArrayVec:
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
        vecshape = get_vecshape(mat)
        entries_impl = self.dense.generate(func_name, mat)
        params = get_parameters(mat)
        entries_impl["matrix"] = array_vector_template(
            func_name,
            params,
            vecshape,
            self.dtype,
            mat.shape,
        )

        output_code = "\n".join(entries_impl.values())

        return watermarked(output_code)
