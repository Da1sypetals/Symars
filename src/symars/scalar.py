import sympy as sp
from .meta import DType, func_template, assert_name, is_constant, CONSTANTS


class GenScalar:
    def __init__(
        self,
        dtype: DType,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
        assert (
            isinstance(precision_digit, int) and precision_digit > 0
        ), f"Precision digit shoud be an unsigned integer, found {precision_digit}"
        assert isinstance(dtype, DType), f"Expected a variant of DType, found {dtype}"
        assert isinstance(tol, float), f"Expected floating point tolerance, found {tol}"

        self.dtype = dtype
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

    def generate_func(self, func_name: str, expr):
        assert_name(func_name)

        params = sorted(list(map(lambda x: str(x), expr.free_symbols)))
        params_decl = [f"{p}: {str(self.dtype)}" for p in params]
        params_list = ", ".join(params_decl)

        return self._generate_func_code(expr, func_name, params_list)

    def generate_func_given_params(self, func_name: str, expr, params):
        """
        You MUST make sure your parameter list is correct!!!
        """
        assert_name(func_name)
        for p in params:
            assert_name(p)

        params_decl = [f"{p}: {str(self.dtype)}" for p in params]
        params_list = ", ".join(params_decl)

        return self._generate_func_code(expr, func_name, params_list)

    def _generate_func_code(self, expr, func_name, params_list):
        code = self.sympy_to_rust(expr)
        const = isinstance(expr, (sp.Number, sp.Integer))

        funcimpl = func_template(
            func_name,
            params_list,
            self.dtype,
            code,
            inline=True,
            const=const,
        )
        return funcimpl

    ###########################################################################
    ########################### main logic entrance ###########################
    ###########################################################################

    def sympy_to_rust(self, expr):
        """Translate a SymPy expression to Rust code."""

        # trigonomics
        if isinstance(expr, sp.sin):
            return f"({self.sympy_to_rust(expr.args[0])}).sin()"
        elif isinstance(expr, sp.cos):
            return f"({self.sympy_to_rust(expr.args[0])}).cos()"
        elif isinstance(expr, sp.tan):
            return f"({self.sympy_to_rust(expr.args[0])}).tan()"
        elif isinstance(expr, sp.cot):
            return f"({self.sympy_to_rust(expr.args[0])}).tan().recip()"
        elif isinstance(expr, sp.asin):
            return f"({self.sympy_to_rust(expr.args[0])}).asin()"
        elif isinstance(expr, sp.acos):
            return f"({self.sympy_to_rust(expr.args[0])}).acos()"
        elif isinstance(expr, sp.atan2):
            # Mind the order here! the order is SAME in SymPy and Rust.
            y = self.sympy_to_rust(expr.args[0])
            x = self.sympy_to_rust(expr.args[1])
            return f"({y}).atan2({x})"

        # hyperbolic trigonomics
        elif isinstance(expr, sp.sinh):
            return f"({self.sympy_to_rust(expr.args[0])}).sinh()"
        elif isinstance(expr, sp.cosh):
            return f"({self.sympy_to_rust(expr.args[0])}).cosh()"
        elif isinstance(expr, sp.tanh):
            return f"({self.sympy_to_rust(expr.args[0])}).tanh()"
        elif isinstance(expr, sp.asinh):
            return f"({self.sympy_to_rust(expr.args[0])}).asinh()"
        elif isinstance(expr, sp.acosh):
            return f"({self.sympy_to_rust(expr.args[0])}).acosh()"
        elif isinstance(expr, sp.atanh):
            return f"({self.sympy_to_rust(expr.args[0])}).atanh()"

        # euler constant related
        elif isinstance(expr, sp.exp):
            return f"({self.sympy_to_rust(expr.args[0])}).exp()"
        elif isinstance(expr, sp.log):
            return f"({self.sympy_to_rust(expr.args[0])}).ln()"

        # other functions
        elif isinstance(expr, sp.sinc):
            arg = self.sympy_to_rust(expr.args[0])
            sinc_nonzero = f"((({arg}).sin()) / ({arg}))"
            return f"(if {self.is_zero_boolean(arg)} {{ {1.0}{self.dtype.suffix()} }} else {{ {sinc_nonzero} }})"

        # discrete and nondifferentiable
        elif isinstance(expr, sp.floor):
            return f"({self.sympy_to_rust(expr.args[0])}).floor()"
        elif isinstance(expr, sp.ceiling):
            return f"({self.sympy_to_rust(expr.args[0])}).ceil()"
        elif isinstance(expr, sp.sign):
            expr_str = f"{self.sympy_to_rust(expr.args[0])}"
            return f"if {self.is_zero_boolean(expr_str)} {{ {expr_str} }} else {{ ({expr_str}).signum() }}"
        elif isinstance(expr, sp.Abs):
            return f"({self.sympy_to_rust(expr.args[0])}).abs()"

        # min / max
        elif isinstance(expr, sp.Min):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            return f"({self.sympy_to_rust(expr.args[0])}).min({self.sympy_to_rust(expr.args[1])})"
        elif isinstance(expr, sp.Max):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            return f"({self.sympy_to_rust(expr.args[0])}).max({self.sympy_to_rust(expr.args[1])})"

        # operators
        elif isinstance(expr, sp.Add):
            operands = [f"({self.sympy_to_rust(arg)})" for arg in expr.args]
            return f'({" + ".join(operands)})'
        elif isinstance(expr, sp.Mul):
            if expr.args[0] == -1:
                val = self.sympy_to_rust(sp.Mul(*(expr.args[1:])))
                return f"(-({val}))"

            operands = [f"({self.sympy_to_rust(arg)})" for arg in expr.args]
            return f'({" * ".join(operands)})'
        elif isinstance(expr, sp.Pow):
            # Check if the exponent is an integer
            base = self.sympy_to_rust(expr.args[0])
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

                return f"({base}).powf({self.sympy_to_rust(exponent)})"
        elif is_constant(expr):
            # For symbols and literals
            return self.parse_constant(expr)
        else:
            raise ValueError(f"Unsupported expression type: {expr}")
