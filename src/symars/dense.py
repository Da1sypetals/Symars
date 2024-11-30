import sympy as sp
from .meta import DType, funcname, assert_name, get_parameters
from .scalar import GenScalar
from sortedcontainers import SortedDict


class GenDense:
    def __init__(
        self,
        dtype: DType,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
        self.dtype = dtype
        self.gen_scalar = GenScalar(dtype, tol, precision_digit, debug)

    def generate(self, func_name: str, mat: sp.Matrix):
        assert_name(func_name)

        m, n = mat.shape
        params = get_parameters(mat)
        entries = SortedDict()
        for mi in range(m):
            for ni in range(n):
                name = funcname(func_name, mi, ni)
                funcimpl = self.gen_scalar.generate_func_given_params(
                    name, mat[mi, ni], params
                )
                entries[str((mi, ni))] = funcimpl

        return entries
