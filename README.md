# Symars
Generate Rust code from symbolic vector and matrix expressions.

# Requirements
```
sympy
```
# Supported
1. Build dense small matrix from dense matrix expr;
2. Build triplet list from triplet-represented sparse matrix expr.

## Supported linear algebra crate 
- `nalgebra`
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