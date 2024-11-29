# Symars
Generate Rust code from symbolic vector and matrix expressions.
Build dense small matrix from symbolic dense matrix expressions.

# Requirements
```
sympy
```

# Use
- `pip install .`
- See `examples.ipynb`.
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
sin, cos, tan, asin, acos, atan2
sinh, cosh, tanh, asinh, acosh, atanh
exp, exp2
floor, ceil // behavior of frac is different in rust and sympy.
ln // sp.log(x)
log // sp.log(x, base)
min, max // sp.Min, sp.Max; note to use only for normal numbers (not inf, nan)
signum // sp.sign

```