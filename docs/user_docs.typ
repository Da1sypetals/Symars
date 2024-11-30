// - Configurations -
#set page(
  paper: "us-letter",
  numbering: "1",
)
#set par(justify: true)
#set heading(numbering: "1.")

#set text(
  font: (
    "Libertinus Serif",
  ),
  size: 12pt,
)

#set text(top-edge: 0.7em, bottom-edge: -0.3em)
#set par(leading: 1em)

// - Configurations -


_This documentation is mostly generated by ChatGPT._
#heading(numbering: none)[Symars Documentation]


= Enum `DType`
- *Description:* Specifies the numeric type (`f32` or `f64`) used for computations in the generated Rust code.

= class `GenScalar`
- *Description:* Generates Rust functions for scalar SymPy expressions.
== Constructor:
```py

def __init__(
        self,
        dtype: DType,
        tol: float = 1e-9,
        precision_digit: int = 20,
        debug: bool = False,
    ):
```
  - `dtype` (`DType`): A `DType` instance specifying the numeric type (`f32` or `f64`).
  - `tol` (`float`, optional): Tolerance for float comparisons. Default: `1e-9`.
  - `precision_digit` (`int`, optional): Number of digits to keep when evaluating constants like $pi$ or $gamma=lim_(n arrow.r infinity) (sum_(k=1)^n 1/k - ln n$). Default: `20`.
  - `debug` (`bool`, optional): If `True`, enables debug output. Default: `False`.

== *Public Methods:*
```py
generate_func(func_name: str, expr: sympy.Expr) -> str
```
- `func_name` (`str`): The name of the generated Rust function.
- `expr` (`sympy.Expr`): A scalar SymPy expression.
- *Returns:* A string containing the generated Rust function.

```py
generate_func_given_params(func_name: str, expr: sympy.Expr, params: List[str]) -> str
```
  - `func_name` (`str`): The name of the generated Rust function.
  - `expr` (`sympy.Expr`): A scalar SymPy expression.
  - `params` (`List[str]`): A list of parameter names for the Rust function.
  - *Returns:* A string containing the generated Rust function.

= class `GenNalgebra`
- *Description:* Generates Rust functions for SymPy matrices using the `nalgebra` crate.
== Constructor:
  - Same as `GenScalar`.

== Public Methods:
```py
generate(func_name: str, mat: sympy.Matrix) -> str
```
- *Description:* Generates a Rust function for the matrix compatible with `nalgebra::SMatrix`.
- `mat` (`sympy.Matrix`): The SymPy matrix to generate code for.
- `func_name` (`str`): The name of the generated Rust function.
- *Returns:* A string containing the generated Rust function.

= class `GenArrayVec`
- *Description:* Generates Rust functions for array-based vector representations.
== Constructor: 
  - Same as `GenScalar`.

== Public Methods:
```py
generate(func_name: str, mat: sympy.Matrix) -> str
```
- *Description:* Generates Rust code to store the matrix as a flattened vector.
- `mat` (`sympy.Matrix`): The SymPy matrix to generate code for.
- `func_name` (`str`): The name of the generated Rust function.
- *Returns:* A string containing the generated Rust code.

= class `GenFaer`
- *Description:* Generates Rust functions for SymPy matrices using the `faer` crate.
== Constructor:
  - Same as `GenScalar`.

== Public Methods:
```py
generate(func_name: str, mat: sympy.Matrix) -> str
```
  - *Description:* Generates a Rust function for the matrix compatible with `faer::MatMut`.
  - `mat` (`sympy.Matrix`): The SymPy matrix to generate code for.
  - `func_name` (`str`): The name of the generated Rust function.
  - *Returns:* A string containing the generated Rust function.

= class `GenFaerVec`
- *Description:* Generates Rust functions for SymPy vectors using the `faer` crate.
  - Note: `faer::Col`, `faer::Row`, and `faer::Mat` are distinct types.
== Constructor: 
  - Same as `GenScalar`.

== Public Methods:
  - `generate(func_name: str, mat: sympy.Matrix) -> str`
    - *Description:* Generates Rust code for SymPy vector representations.
    - `mat` (`sympy.Matrix`): The SymPy matrix or vector to generate code for.
    - `func_name` (`str`): The name of the generated Rust function.
    - *Returns:* A string containing the generated Rust code.

= class `GenSparse`
- *Description:* Generates Rust functions for triplet representations of sparse matrices.
== Constructor: 
  - Same as `GenScalar`.

== Public Methods:
```py
generate(exprs: list[sympy.Expr], func_name: str) -> str
```
- *Description:* Generates Rust functions for sparse representations.
- `mat` (`sympy.Matrix`): The SymPy matrix to generate code for.
- `func_name` (`str`): The name of the generated Rust function.
- *Returns:* A string containing the generated Rust code.

= class `GenDense`
- *Description:* Generates Rust functions for dense matrices. *Not user-facing; inspect only for debugging purposes.*
== Constructor:
  - Same as `GenScalar`.

== Public Methods:
```py
generate(func_name: str, mat: sympy.Matrix) -> str
```
- *Description:* Generates Rust functions to represent the entries of a dense matrix.
- `mat` (`sympy.Matrix`): The SymPy matrix to generate code for.
- `func_name` (`str`): The name of the generated Rust function.
- *Returns:* A string containing the generated Rust function.

#let headless(x) = heading(numbering: none)[#x]

#headless[Appendix: Semantics]

#headless[`sp.sign`]
`sp.sign` is implemented to return *itself* with input `+0.0` *and* `-0.0`. 

Its semantics is preserved in Symars for the sake of correctness, as some function has sign function in their derivatives. For example, it generates 
```rust
if x.abs() == 0.0_f64 { x } else { x.signum() }
      
```
rather than `x.signum()`.