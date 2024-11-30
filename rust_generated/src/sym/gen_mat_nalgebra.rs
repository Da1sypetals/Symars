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
    ((-(((((((d) * ((e).recip())) + (((-(h)) + ((f) * (g))).ln())).abs()).cbrt())
        * ((a) + (b) + (-(c)))
        * ((c).cosh()))
    .tanh()))
        + (((((b).sqrt()).exp()) + (((a).ln()).sin())).abs()))
}

#[inline]
pub fn test_matrix_0_1(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-((f).tanh())) + ((e).cosh()) + ((d).sinh()))
}

#[inline]
pub fn test_matrix_0_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((((a).ln()).sin())
        + (if (((c).powf(((0.420000000000000_f64) * ((d).powf(3.1415926535897932385_f64)))))
            + (((b).sqrt()).exp()))
        .abs()
            == 0.0_f64
        {
            1.0_f64
        } else {
            (((((c).powf(((0.420000000000000_f64) * ((d).powf(3.1415926535897932385_f64)))))
                + (((b).sqrt()).exp()))
            .sin())
                / (((c).powf(((0.420000000000000_f64) * ((d).powf(3.1415926535897932385_f64)))))
                    + (((b).sqrt()).exp())))
        }))
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
    1.0000000000000000000_f64
}

#[inline]
pub fn test_matrix_1_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((((b).cos()) * ((c).tan())) + ((a).sin()))
}

#[inline]
pub fn test_matrix_2_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-(((3.1415926535897932385_f64)
        * (((((d) * ((e).recip())) + (((-(h)) + ((f) * (g))).ln())).abs()).cbrt())
        * ((c).cosh())
        * (if ((0.473417601666430_f64) * (a) * ((a) + (b) + (-(c)))).abs() == 0.0_f64 {
            1.0_f64
        } else {
            ((((0.473417601666430_f64) * (a) * ((a) + (b) + (-(c)))).sin())
                / ((0.473417601666430_f64) * (a) * ((a) + (b) + (-(c)))))
        }))
    .tanh()))
        + ((((((b)
            + (((c) + (-(a)))
                .powf(((0.33333333333333333333_f64) + (1.8392867552141611326_f64)))))
        .sqrt())
        .exp())
            + (((0.60000000000000000000_f64)
                + ((4.2307692307692307692_f64).powf(2.7182818284590452354_f64))
                + (((c) + (-(a))).powf(0.25000000000000000000_f64))
                + (-(0.57721566490153286061_f64))
                + (((14.1444000000000_f64) * (a)).ln()))
            .sin()))
        .abs()))
}

#[inline]
pub fn test_matrix_2_1(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-3.0000000000000000000_f64)
        + ((2.0000000000000000000_f64) * (b))
        + ((3.50000000000000_f64) * (a)))
}

#[inline]
pub fn test_matrix_2_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    ((-(((a) + (h)).exp()))
        + ((d) * ((e).recip()) * ((a) + (b) + (-(c))))
        + (((h) + ((f) * (g))).ln()))
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
