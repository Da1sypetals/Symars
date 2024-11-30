/*

* Code generated by Symars. Thank you for using Symars!
  Symars is licensed under MIT licnese.
  Repository: https://github.com/Da1sypetals/Symars

* Computation code is not intended for manual editing.

* If you find an error,
  or if you believe Symars generates incorrect result,
  please raise an issue under our repo with minimal reproducible example.

*/

/*

    value at index position 0 = exprs_0
    value at index position 1 = exprs_1
    value at index position 2 = exprs_2
    value at index position 3 = exprs_3
    value at index position 4 = exprs_4
    value at index position 5 = exprs_5
    value at index position 6 = exprs_6
    value at index position 7 = exprs_7
    value at index position 8 = exprs_8
    value at index position 9 = exprs_9

*/

#[inline]
pub fn exprs_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    (((d).powf(((e) + (f)))) + (((((a) + (b)).powi(2)) * ((c).exp())).sqrt()))
}

#[inline]
pub fn exprs_1(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    (((c).max(d)) + ((a).min(b)))
}

#[inline]
pub fn exprs_2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    ((((b).cos()) * ((c).tan())) + ((a).sin()))
}

#[inline]
pub fn exprs_3(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    ((-3.0000000000000000000_f64)
        + ((2.0000000000000000000_f64) * (b))
        + ((3.50000000000000_f64) * (a)))
}

#[inline]
pub fn exprs_4(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    (((a).powi(2)) + ((b).sqrt()) + ((c).recip()))
}

#[inline]
pub fn exprs_5(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    ((((c) + (-(d))).ceil()) + (((a) + (b)).floor()))
}

#[inline]
pub fn exprs_6(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    ((-(((c).cosh()).tanh())) + (((b).sqrt()).exp()) + (((a).ln()).sin()))
}

#[inline]
pub fn exprs_7(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    ((((b).exp()) * ((a).ln())) + (((c) + (d)).ln()))
}

#[inline]
pub fn exprs_8(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    ((-(if ((c) + (d)).abs() == 0.0_f64 {
        ((c) + (d))
    } else {
        ((c) + (d)).signum()
    })) + (if ((a) + (-(b))).abs() == 0.0_f64 {
        ((a) + (-(b)))
    } else {
        ((a) + (-(b))).signum()
    }))
}

#[inline]
pub const fn exprs_9(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    0_f64
}

pub fn exprs(
    triplets: &mut Vec<(usize, usize, f64)>,
    indices: &[(usize, usize)],
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
) {
    triplets.push((indices[0].0, indices[0].1, exprs_0(a, b, c, d, e, f)));

    triplets.push((indices[1].0, indices[1].1, exprs_1(a, b, c, d, e, f)));

    triplets.push((indices[2].0, indices[2].1, exprs_2(a, b, c, d, e, f)));

    triplets.push((indices[3].0, indices[3].1, exprs_3(a, b, c, d, e, f)));

    triplets.push((indices[4].0, indices[4].1, exprs_4(a, b, c, d, e, f)));

    triplets.push((indices[5].0, indices[5].1, exprs_5(a, b, c, d, e, f)));

    triplets.push((indices[6].0, indices[6].1, exprs_6(a, b, c, d, e, f)));

    triplets.push((indices[7].0, indices[7].1, exprs_7(a, b, c, d, e, f)));

    triplets.push((indices[8].0, indices[8].1, exprs_8(a, b, c, d, e, f)));

    triplets.push((indices[9].0, indices[9].1, exprs_9(a, b, c, d, e, f)));
}
