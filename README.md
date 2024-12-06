# Symars
Generate Rust code from SymPy expressions, including matrices.

# Use
- Clone this repo and `pip install .`
- See `examples.ipynb` for example use.
- Refer to `docs/user_docs.pdf` for details.
- See `rust_generated/src/sym/*.rs` for sample generated code.

# Support

## Supported linear algebra crate 
- `nalgebra`, returns a matrix (including row and column vectors, which share the underlying type `nalgebra::SMatrix`)
- `faer`, passes in a mutable slice of matrix and modifies it.
  - Both `Row`, `Col` and `Mat` are supported.
- Built-in array `[f32; N]` and `[f64; N]` for vector.
- Sparse matrix with its triplet representation stored in `Vec<(usize, usize, value_type)>`.

> If you want to support your own linear algebra crate, refer to `nalgebra.py`.

## Supported data types
`f32`, `f64`

## Supported operations
```rust
+ - * /
powf, sqrt
powi // only for constant integers
sin, cos, tan, cot, asin, acos, atan2
sinh, cosh, tanh, asinh, acosh, atanh
exp, exp2
floor, ceil // behavior of frac is different in rust and sympy.
ln // sp.log(x)
log // sp.log(x, base)
min, max // sp.Min, sp.Max; note to use only for normal numbers (not inf, nan)
signum // sp.sign
abs

```


# Dependencies
```
sympy
sortedcontainers
```


# Note
1. Some functions are ambiguous (for example, `sign`). Please refer to documentation for their semantics.
2. Please handle NaN and inf carefully by yourself.
3. We don't do indent/format. `cargo fmt` do it for us.
4. Generated function has parameters in lexicographical order of identifiers (default in `sortedcontainers`).
5. If you are using rust-analyzer VSCode extension, try to put your numerical computation code in a crate different from your driver code, especially when the code exceeds say, 10000 lines. Otherwise, your rust-analyzer may hang often.