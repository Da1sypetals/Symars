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
from .vecshape import VecShape, get_vecshape


def faer_matrix_template(name, params, dtype, return_shape):
    assert_name(name)
    assert (
        len(return_shape) == 2
    ), "Return shape shoule have 2 dimensions, found {return_shape}"

    m, n = return_shape
    range_prod = itertools.product(range(m), range(n))
    param_list = ", ".join([f"{p}: {str(dtype)}" for p in params])
    param_invoke = ", ".join(params)

    matmut_type = f"faer::MatMut<{str(dtype)}>"

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


class GenFaer:
    def __init__(
        self,
        dtype: DType,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
        self.dtype = dtype
        self.dense = GenDense(dtype, tol, precision_digit, debug)

    def generate(self, mat: sp.Matrix, func_name: str):
        entries_impl = self.dense.generate(mat, func_name)
        params = get_parameters(mat)
        entries_impl["matrix"] = faer_matrix_template(
            func_name, params, self.dtype, mat.shape
        )

        output_code = "\n".join(entries_impl.values())

        return watermarked(output_code)


##############################################
################### vector ###################
##############################################


def faer_vector_template(name, params, vecshape: VecShape, dtype, return_shape):
    assert_name(name)
    assert (
        len(return_shape) == 2
    ), "Return shape shoule have 2 dimensions, found {return_shape}"

    m, n = return_shape
    param_list = ", ".join([f"{p}: {str(dtype)}" for p in params])
    param_invoke = ", ".join(params)

    vecmut_type = f"faer::{str(vecshape)}Mut<{str(dtype)}>"

    def entry_assign(i):
        return f"""
vec[{i}] = {vecshape.func_name(name, i)}({param_invoke});
"""

    assigns = "\n".join([entry_assign(i) for i in range(m * n)])
    return f"""
pub fn {name}(mut vec: {vecmut_type}, {param_list}) {{

    {assigns}
    
}}
"""


class GenFaerVec:
    def __init__(
        self,
        dtype: DType,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
        self.dtype = dtype
        self.dense = GenDense(dtype, tol, precision_digit, debug)

    def generate(self, mat: sp.Matrix, func_name: str):
        vecshape = get_vecshape(mat)
        entries_impl = self.dense.generate(mat, func_name)
        params = get_parameters(mat)
        entries_impl["matrix"] = faer_vector_template(
            func_name,
            params,
            vecshape,
            self.dtype,
            mat.shape,
        )

        output_code = "\n".join(entries_impl.values())

        return watermarked(output_code)
