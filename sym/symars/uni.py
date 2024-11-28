import sympy as sp
from .meta import DType, func_template


###################


class SymarsUni:
    def __init__(self, dtype: DType, tol: float = 1e-8, debug: bool = False):
        self.dtype = dtype
        self.debug_on = debug
        self.tol = tol

    def float_eq(self, a, b):
        return sp.Abs(a - b) < self.tol

    def debug(self, *args, **kw):
        if self.debug_on:
            print("[symars debug] ", end="")
            print(*args, **kw)

    @property
    def literal_suffix(self):
        """Return the appropriate suffix for the type (either F32 or F64)."""
        if self.dtype == DType.F32:
            return "_f32"
        else:
            return "_f64"

    @property
    def param_type(self):
        if self.dtype == DType.F32:
            return "f32"
        else:
            return "f64"

    def parse_symbol_or_literal(self, expr):
        """Parse the input and ensure it's either a symbol or a literal (int or float)."""
        if isinstance(expr, sp.Symbol):
            # It's a symbol, return its name (Rust variable)
            self.debug(f"symbol: {expr}")
            return str(expr)
        elif isinstance(expr, (int, float, sp.Integer, sp.Number)):
            # It's a literal, return with the correct suffix.
            # convert all to float.
            self.debug(f"literal: {expr}")
            self.debug(f"{expr}{self.literal_suffix}")
            return f"{expr}{self.literal_suffix}"
        else:
            # Raise an error if neither symbol nor literal
            raise ValueError(
                f"Invalid expression: {expr}. Must be a symbol or a literal."
            )

    def generate_func(self, name: str, expr):
        params = sorted(list(map(lambda x: str(x), expr.free_symbols)))
        params_decl = [f"{p}: {self.param_type}" for p in params]
        params_list = ", ".join(params_decl)

        code = self.sympy_to_rust(expr)

        funcimpl = func_template(name, params_list, self.param_type, code, inline=True)
        return funcimpl

    def generate_func_given_params(self, name: str, expr, params):
        """
        You MUST make sure your parameter list is correct!!!
        """
        params_decl = [f"{p}: {self.param_type}" for p in params]
        params_list = ", ".join(params_decl)

        code = self.sympy_to_rust(expr)

        funcimpl = func_template(name, params_list, self.param_type, code, inline=True)
        return funcimpl

    def sympy_to_rust(self, expr):
        """Translate a SymPy expression to Rust code."""
        if isinstance(expr, sp.sin):
            return f"({self.sympy_to_rust(expr.args[0])}).sin()"
        elif isinstance(expr, sp.cos):
            return f"({self.sympy_to_rust(expr.args[0])}).cos()"
        elif isinstance(expr, sp.tan):
            return f"({self.sympy_to_rust(expr.args[0])}).tan()"
        elif isinstance(expr, sp.asin):
            return f"({self.sympy_to_rust(expr.args[0])}).asin()"
        elif isinstance(expr, sp.acos):
            return f"({self.sympy_to_rust(expr.args[0])}).acos()"
        elif isinstance(expr, sp.atan2):
            return f"({self.sympy_to_rust(expr.args[1])}).atan2({self.sympy_to_rust(expr.args[0])})"
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
        elif isinstance(expr, sp.exp):
            return f"({self.sympy_to_rust(expr.args[0])}).exp()"
        elif isinstance(expr, sp.floor):
            return f"({self.sympy_to_rust(expr.args[0])}).floor()"
        elif isinstance(expr, sp.ceiling):
            return f"({self.sympy_to_rust(expr.args[0])}).ceil()"
        elif isinstance(expr, sp.log):
            return f"({self.sympy_to_rust(expr.args[0])}).ln()"
        elif isinstance(expr, sp.Min):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            return f"({self.sympy_to_rust(expr.args[0])}).min({self.sympy_to_rust(expr.args[1])})"
        elif isinstance(expr, sp.Max):
            if len(expr.args) != 2:
                raise ValueError("Min and Max should have 2 arguments!")
            return f"({self.sympy_to_rust(expr.args[0])}).max({self.sympy_to_rust(expr.args[1])})"
        elif isinstance(expr, sp.sign):
            return f"({self.sympy_to_rust(expr.args[0])}).signum()"
        elif isinstance(expr, sp.Add):
            operands = [f"({self.sympy_to_rust(arg)})" for arg in expr.args]
            return " + ".join(operands)
        elif isinstance(expr, sp.Mul):
            operands = [f"({self.sympy_to_rust(arg)})" for arg in expr.args]
            return " * ".join(operands)
        elif isinstance(expr, sp.Pow):
            # Check if the exponent is an integer
            base = self.sympy_to_rust(expr.args[0])
            exponent = expr.args[1]
            if isinstance(exponent, sp.Integer):
                if exponent == 1:
                    return f"({base})"
                if exponent == -1:
                    return f"({base}).recip()"
                return f"({base}).powi({exponent})"
            else:
                if isinstance(expr.args[1], sp.core.numbers.Half):
                    return f"({base}).sqrt()"

                if isinstance(exponent, sp.Number) and self.float_eq(exponent, 0.5):
                    return f"({base}).sqrt()"

                return f"({base}).powf({self.sympy_to_rust(exponent)})"
        elif isinstance(expr, (sp.Symbol, int, float, sp.Integer, sp.Number)):
            # For symbols and literals
            return self.parse_symbol_or_literal(expr)
        else:
            raise ValueError(f"Invalid expression type: {expr}")
