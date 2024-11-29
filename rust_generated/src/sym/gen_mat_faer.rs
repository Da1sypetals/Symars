/*

* Code generated by Symars. Thank you for using Symars!
  Symars is licensed under MIT licnese.
  Repository: https://github.com/Da1sypetals/Symars

* Computation code is not intended for manual editing.

* If you find an error,
  or if you believe Symars generates incorrect result,
  please raise an issue under our repo with minimal reproducible example.

*/

#[inline]
pub fn test_matrix_0_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    (((b).exp()) * ((a).ln())) + (((c) + (d)).ln())
}

#[inline]
pub fn test_matrix_0_1(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    (-3_f64) + ((2_f64) * (b)) + ((3.50000000000000_f64) * (a))
}

#[inline]
pub fn test_matrix_0_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((d).powf((e) + (f))) + (((((a) + (b)).powi(2)) * ((c).exp())).sqrt())
}

#[inline]
pub const fn test_matrix_1_0(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
) -> f64 {
    0_f64
}

#[inline]
pub const fn test_matrix_1_1(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
) -> f64 {
    1_f64
}

#[inline]
pub fn test_matrix_1_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-1_f64) * ((f).tanh())) + ((e).cosh()) + ((d).sinh())
}

#[inline]
pub fn test_matrix_2_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    (((c) + ((-1_f64) * (d))).ceil()) + (((a) + (b)).floor())
}

#[inline]
pub fn test_matrix_2_1(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-1_f64) * ((d).atan2(c))) + ((b).acos()) + ((a).asin())
}

#[inline]
pub fn test_matrix_2_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-1_f64) * (((a) + (h)).exp()))
        + ((d) * ((e).recip()) * ((a) + (b) + ((-1_f64) * (c))))
        + ((((-1_f64) * (h)) + ((f) * (g))).ln())
}

pub fn test_matrix(
    mut mat: faer::MatMut<f64>,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
) {
    mat[(0, 0)] = test_matrix_0_0(a, b, c, d, e, f, g, h);

    mat[(0, 1)] = test_matrix_0_1(a, b, c, d, e, f, g, h);

    mat[(0, 2)] = test_matrix_0_2(a, b, c, d, e, f, g, h);

    mat[(1, 0)] = test_matrix_1_0(a, b, c, d, e, f, g, h);

    mat[(1, 1)] = test_matrix_1_1(a, b, c, d, e, f, g, h);

    mat[(1, 2)] = test_matrix_1_2(a, b, c, d, e, f, g, h);

    mat[(2, 0)] = test_matrix_2_0(a, b, c, d, e, f, g, h);

    mat[(2, 1)] = test_matrix_2_1(a, b, c, d, e, f, g, h);

    mat[(2, 2)] = test_matrix_2_2(a, b, c, d, e, f, g, h);
}
