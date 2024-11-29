import sympy as sp
from .meta import DType, funcname, assert_name
from .uni import SymarsUni


class SymarsDense:
    def __init__(self, dtype: DType, tol: float = 1e-9, debug: bool = False):
        self.dtype = dtype
        self.uni = SymarsUni(dtype, tol, debug)

    def params(self, mat: sp.Matrix):
        return sorted(list(map(lambda x: str(x), mat.free_symbols)))

    def generate(self, mat: sp.Matrix, func_name: str):
        assert_name(func_name)

        m, n = mat.shape
        params = self.params(mat)
        entries = {}
        for mi in range(m):
            for ni in range(n):
                name = funcname(func_name, mi, ni)
                funcimpl = self.uni.generate_func_given_params(
                    name, mat[mi, ni], params
                )
                entries[(mi, ni)] = funcimpl

        return entries
