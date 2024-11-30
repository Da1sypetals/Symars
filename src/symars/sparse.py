import sympy as sp
from .meta import DType, funcname_vector, assert_name, watermarked, get_parameters
from .scalar import GenScalar
from sortedcontainers import SortedDict


class GenSparse:
    def __init__(
        self,
        dtype: DType,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
        self.dtype = dtype
        self.gen_scalar = GenScalar(dtype, tol, precision_digit, debug)

    def params(self, mat: sp.Matrix):
        return sorted(list(map(lambda x: str(x), mat.free_symbols)))

    def _generate_comments(self, num_expr: int, func_name: str):
        def comment_item(i):
            return f"value at index position {i} = {funcname_vector(func_name, i)}"

        comment_content = "\n\t".join([comment_item(i) for i in range(num_expr)])
        return f"/*\n\n\t{comment_content}\n\n*/\n\n"

    def _generate_entries_code(self, exprs: list[sp.Expr], func_name: str):
        assert_name(func_name)

        num_expr = len(exprs)
        params = get_parameters(exprs)

        impls = SortedDict(
            {
                str(i): self.gen_scalar.generate_func_given_params(
                    funcname_vector(func_name, i), exprs[i], params
                )
                for i in range(num_expr)
            }
        )

        return impls

    def _generate_triplets_code(self, exprs: list[sp.Expr], func_name: str):
        num_expr = len(exprs)
        params = get_parameters(exprs)
        param_list = ", ".join([f"{p}: {str(self.dtype)}" for p in params])
        param_invoke = ", ".join(params)

        triplet_type = f"&mut Vec<(usize, usize, {str(self.dtype)})>"

        def entry_assign(i):
            return f"""
    triplets.push((indices[{i}].0, indices[{i}].1, {funcname_vector(func_name, i)}({param_invoke})));
"""

        assigns = "\n".join([entry_assign(i) for i in range(num_expr)])

        triplet_code = f"""
pub fn {func_name}(triplets: {triplet_type}, indices: &[(usize, usize)], {param_list}) {{

    {assigns}

}}
"""
        return triplet_code

    def generate(self, exprs: list[sp.Expr], func_name: str):
        entries = self._generate_entries_code(exprs, func_name)
        triplets = self._generate_triplets_code(exprs, func_name)
        entries["triplets"] = triplets

        comments = self._generate_comments(len(exprs), func_name)

        output_code = comments + "\n".join(entries.values())

        return watermarked(output_code)
