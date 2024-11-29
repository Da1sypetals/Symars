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
    (((d).exp()).max((e).cos())) + (((c).sin()).min(((a) + (b)).ln()))
}

#[inline]
pub fn test_matrix_0_1(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((d).powf((e) + (f))) + (((((a) + (b)).powi(2)) * ((c).exp())).sqrt())
}

#[inline]
pub fn test_matrix_0_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-1_f64) * (((c) + (d)).signum())) + (((a) + ((-1_f64) * (b))).signum())
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
    ((a).powi(2)) + ((b).sqrt()) + ((c).recip())
}

#[inline]
pub fn test_matrix_2_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((c).max(d)) + ((a).min(b))
}

#[inline]
pub fn test_matrix_2_1(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-1_f64) * (((a) + (h)).exp()))
        + ((d) * ((e).recip()) * ((a) + (b) + ((-1_f64) * (c))))
        + ((((-1_f64) * (h)) + ((f) * (g))).ln())
}

#[inline]
pub fn test_matrix_2_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    (((b).exp()) * ((a).ln())) + (((c) + (d)).ln())
}

pub fn test_matrix(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
) -> nalgebra::SMatrix<f64, 3, 3> {
    let mut result = nalgebra::SMatrix::zeros();

    result[(0, 0)] = test_matrix_0_0(a, b, c, d, e, f, g, h);

    result[(0, 1)] = test_matrix_0_1(a, b, c, d, e, f, g, h);

    result[(0, 2)] = test_matrix_0_2(a, b, c, d, e, f, g, h);

    result[(1, 0)] = test_matrix_1_0(a, b, c, d, e, f, g, h);

    result[(1, 1)] = test_matrix_1_1(a, b, c, d, e, f, g, h);

    result[(1, 2)] = test_matrix_1_2(a, b, c, d, e, f, g, h);

    result[(2, 0)] = test_matrix_2_0(a, b, c, d, e, f, g, h);

    result[(2, 1)] = test_matrix_2_1(a, b, c, d, e, f, g, h);

    result[(2, 2)] = test_matrix_2_2(a, b, c, d, e, f, g, h);

    result
}
