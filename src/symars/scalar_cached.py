import sympy as sp
from .meta import (
    DType,
    assert_name,
    is_constant,
    CONSTANTS,
    get_parameters,
    watermarked,
)


def cached_func_template(
    name,
    temps,
    param_list,
    value_type: DType,
    code,
    inline: bool = True,
    const: bool = False,
):
    temp_defs = "\n".join(temps)

    code = f"""

{"#[inline]" if inline else ""}
pub {"const" if const else ""} fn {name}({param_list}) -> {str(value_type)} {{

    {temp_defs}

    {code}

}}
"""
    return watermarked(code)


class Subtree:
    def __init__(self, name):
        self.name = name
        # detected count in an expression tree
        self.count = 0
        # use count as a tree root node
        self.use_count = 0
        # is used as a tree root node
        self.used = False

    def incr(self):
        self.count += 1

    def count_used(self):
        self.use_count += 1

    def use(self):
        self.used = True


class GenScalarCached:
    def __init__(
        self,
        dtype: DType,
        min_occurrence: int = 2,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
        assert (
            isinstance(min_occurrence, int) and min_occurrence >= 2
        ), "Minimum cached occurrences should be an integer and >= 2. Otherwise use symars.GenScalar instead."
        assert (
            isinstance(precision_digit, int) and precision_digit > 0
        ), f"Precision digit shoud be an unsigned integer, found {precision_digit}"
        assert isinstance(dtype, DType), f"Expected a variant of DType, found {dtype}"
        assert isinstance(tol, float), f"Expected floating point tolerance, found {tol}"

        self.dtype = dtype
        self.min_occurrence = min_occurrence
        self.debug_on = debug
        self.tol = tol
        self.precision_digit = precision_digit

    def float_eq(self, a, b):
        return sp.Abs(a - b) < self.tol

    def is_zero_boolean(self, expr_str: str):
        return f"({expr_str}).abs() == 0.0{self.dtype.suffix()}"

    def debug(self, *args, **kw):
        if self.debug_on:
            print("[symars debug] ", end="")
            print(*args, **kw)

    def parse_constant(self, expr):
        """Parse the input and ensure it's either a symbol or a literal (int or float)."""
        # the most specific ones: constants
        if expr in CONSTANTS:
            return f"{expr.evalf(self.precision_digit)}{self.dtype.suffix()}"
        elif isinstance(expr, sp.Symbol):
            # It's a symbol, return its name (Rust variable)
            self.debug(f"symbol: {expr}")
            return str(expr)
        elif isinstance(expr, sp.Rational):
            self.debug(f"{expr.evalf(self.precision_digit)}{self.dtype.suffix()}")
            return f"{expr.evalf(self.precision_digit)}{self.dtype.suffix()}"
        elif isinstance(expr, (int, float, sp.Integer, sp.Float)):
            # It's a literal, return with the correct suffix.
            # convert all to float.
            self.debug(f"literal: {expr}")
            self.debug(f"{expr}{self.dtype.suffix()}")
            return f"{expr}{self.dtype.suffix()}"

        else:
            # Raise an error if neither symbol nor literal
            raise ValueError(
                f"Invalid constant expression: {expr}. Must be a symbol, literal or Rational."
            )

    def generate(self, func_name: str, expr: sp.Expr):
        assert_name(func_name)

        params = get_parameters(expr)
        params_decl = [f"{p}: {str(self.dtype)}" for p in params]
        params_list = ", ".join(params_decl)

        return self._generate_func_code(expr, func_name, params_list)

    def generate_func_given_params(
        self, func_name: str, expr: sp.Expr, params: list[str]
    ):
        """
        You MUST make sure your parameter list is correct!!!
        """
        assert_name(func_name)
        for p in params:
            assert_name(p)

        params_decl = [f"{p}: {str(self.dtype)}" for p in params]
        params_list = ", ".join(params_decl)

        return self._generate_func_code(expr, func_name, params_list)

    def _generate_subtree_code(self, expr: sp.Expr, subtree: Subtree):
        # Actual code computing the result of subtree
        code = self.sympy_to_rust(expr, {})
        return f"let {subtree.name} = {code};"

    def _generate_func_code(self, expr, func_name, params_list):
        cached = self.cache_all_subtree(expr, min_occurrence=self.min_occurrence)
        cached = self.check_usage(expr, self.min_occurrence, cached)

        code = self.sympy_to_rust(expr, cached)
        const = isinstance(expr, (sp.Number, sp.Integer))

        temps = [
            self._generate_subtree_code(expr, subtree)
            for expr, subtree in cached.items()
        ]

        funcimpl = cached_func_template(
            func_name,
            temps,
            params_list,
            self.dtype,
            code,
            inline=True,
            const=const,
        )
        return funcimpl

    def _cache_subtree_recursive(self, expr: sp.Expr, cache):
        """Translate a SymPy expression to Rust code."""

        # trigonomics
        if isinstance(expr, sp.sin):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.cos):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.tan):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.cot):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.asin):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.acos):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.atan2):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
            self._cache_subtree_recursive(expr.args[1], cache)

        # hyperbolic trigonomics
        elif isinstance(expr, sp.sinh):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.cosh):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.tanh):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.asinh):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.acosh):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.atanh):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)

        # euler constant related
        elif isinstance(expr, sp.exp):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.log):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)

        # other functions
        elif isinstance(expr, sp.sinc):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)

        # discrete and nondifferentiable
        elif isinstance(expr, sp.floor):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.ceiling):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.sign):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
        elif isinstance(expr, sp.Abs):
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)

        # min / max
        elif isinstance(expr, sp.Min):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
            self._cache_subtree_recursive(expr.args[1], cache)
        elif isinstance(expr, sp.Max):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            if cache(expr):
                return
            self._cache_subtree_recursive(expr.args[0], cache)
            self._cache_subtree_recursive(expr.args[1], cache)

        # operators
        elif isinstance(expr, sp.Add):
            if cache(expr):
                return
            for operand in expr.args:
                self._cache_subtree_recursive(operand, cache)

        elif isinstance(expr, sp.Mul):
            if cache(expr):
                return
            for operand in expr.args:
                self._cache_subtree_recursive(operand, cache)

        elif isinstance(expr, sp.Pow):
            if cache(expr):
                return
            for operand in expr.args:
                self._cache_subtree_recursive(operand, cache)

        elif is_constant(expr):
            pass
        else:
            raise ValueError(f"Unsupported expression type: {expr}")

    ###########################################################################
    ########################### main logic entrance ###########################
    ###########################################################################

    def cache_all_subtree(self, expr, min_occurrence):
        cached = dict()
        # cached = SortedDict()
        name_counter = 0

        def name_generator():
            nonlocal name_counter
            name = f"__intermediate_result_{name_counter}"
            name_counter += 1
            return name

        def cache(expr: sp.Expr):
            """Return whether recursion should terminate"""
            if expr not in cached:
                name = name_generator()
                cached[expr] = Subtree(name)
            cached[expr].incr()
            return False

        self._cache_subtree_recursive(expr, cache)

        cached = {k: v for k, v in cached.items() if v.count >= min_occurrence}
        return cached

    def check_usage(self, expr, min_occurrence, cached):
        def use_expr(expr: sp.Expr):
            """Return whether recursion should terminate"""
            if expr in cached:
                # print(f"cache: {expr}")
                cached[expr].count_used()
                return True
            return False

        self._cache_subtree_recursive(expr, use_expr)

        cached = {k: v for k, v in cached.items() if v.use_count >= min_occurrence}
        return cached

    def sympy_to_rust(self, expr, cached):
        """Translate a SymPy expression to Rust code."""

        if expr in cached:
            cached[expr].use()
            self.debug(f"!!use: {cached[expr].name}")
            return f"({cached[expr].name})"

        # trigonomics
        if isinstance(expr, sp.sin):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).sin()"
        elif isinstance(expr, sp.cos):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).cos()"
        elif isinstance(expr, sp.tan):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).tan()"
        elif isinstance(expr, sp.cot):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).tan().recip()"
        elif isinstance(expr, sp.asin):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).asin()"
        elif isinstance(expr, sp.acos):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).acos()"
        elif isinstance(expr, sp.atan2):
            # Mind the order here! the order is SAME in SymPy and Rust.
            y = self.sympy_to_rust(expr.args[0], cached)
            x = self.sympy_to_rust(expr.args[1], cached)
            return f"({y}).atan2({x})"

        # hyperbolic trigonomics
        elif isinstance(expr, sp.sinh):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).sinh()"
        elif isinstance(expr, sp.cosh):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).cosh()"
        elif isinstance(expr, sp.tanh):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).tanh()"
        elif isinstance(expr, sp.asinh):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).asinh()"
        elif isinstance(expr, sp.acosh):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).acosh()"
        elif isinstance(expr, sp.atanh):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).atanh()"

        # euler constant related
        elif isinstance(expr, sp.exp):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).exp()"
        elif isinstance(expr, sp.log):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).ln()"

        # other functions
        elif isinstance(expr, sp.sinc):
            arg = self.sympy_to_rust(expr.args[0], cached)
            sinc_nonzero = f"((({arg}).sin()) / ({arg}))"
            return f"(if {self.is_zero_boolean(arg)} {{ {1.0}{self.dtype.suffix()} }} else {{ {sinc_nonzero} }})"

        # discrete and nondifferentiable
        elif isinstance(expr, sp.floor):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).floor()"
        elif isinstance(expr, sp.ceiling):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).ceil()"
        elif isinstance(expr, sp.sign):
            expr_str = f"{self.sympy_to_rust(expr.args[0], cached)}"
            return f"(if {self.is_zero_boolean(expr_str)} {{ {expr_str} }} else {{ ({expr_str}).signum() }})"
        elif isinstance(expr, sp.Abs):
            return f"({self.sympy_to_rust(expr.args[0], cached)}).abs()"

        # min / max
        elif isinstance(expr, sp.Min):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            return f"({self.sympy_to_rust(expr.args[0], cached)}).min({self.sympy_to_rust(expr.args[1], cached)})"
        elif isinstance(expr, sp.Max):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            return f"({self.sympy_to_rust(expr.args[0], cached)}).max({self.sympy_to_rust(expr.args[1], cached)})"

        # operators
        elif isinstance(expr, sp.Add):
            operands = [f"({self.sympy_to_rust(arg, cached)})" for arg in expr.args]
            return f'({" + ".join(operands)})'
        elif isinstance(expr, sp.Mul):
            if expr.args[0] == -1:
                val = self.sympy_to_rust(sp.Mul(*(expr.args[1:])), cached)
                return f"(-({val}))"

            operands = [f"({self.sympy_to_rust(arg, cached)})" for arg in expr.args]
            return f'({" * ".join(operands)})'
        elif isinstance(expr, sp.Pow):
            # Check if the exponent is an integer
            base = self.sympy_to_rust(expr.args[0], cached)
            exponent = expr.args[1]
            if isinstance(exponent, sp.Integer):
                if exponent == 1 or isinstance(exponent, sp.core.numbers.One):
                    return f"({base})"
                if exponent == -1 or isinstance(exponent, sp.core.numbers.NegativeOne):
                    return f"({base}).recip()"
                return f"({base}).powi({exponent})"
            else:
                if isinstance(exponent, sp.core.numbers.Half):
                    return f"({base}).sqrt()"

                if exponent == sp.Rational(1, 2):
                    return f"({base}).sqrt()"
                if exponent == sp.Rational(1, 3):
                    return f"({base}).cbrt()"

                if isinstance(exponent, sp.Number) and self.float_eq(exponent, 0.5):
                    return f"({base}).sqrt()"

                return f"({base}).powf({self.sympy_to_rust(exponent, cached)})"
        elif is_constant(expr):
            # For symbols and literals
            return self.parse_constant(expr)
        else:
            raise ValueError(f"Unsupported expression type: {expr}")
