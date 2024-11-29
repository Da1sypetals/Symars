
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
pub  fn test_vector_0_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    ((c).max(d)) + ((a).min(b))

}




#[inline]
pub  fn test_vector_1_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    (-3_f64) + ((2_f64) * (b)) + ((3.50000000000000_f64) * (a))

}




#[inline]
pub  fn test_vector_2_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    (((b).cos()) * ((c).tan())) + ((a).sin())

}




#[inline]
pub const fn test_vector_3_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    0_f64

}




#[inline]
pub const fn test_vector_4_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    1_f64

}




#[inline]
pub  fn test_vector_5_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    ((-1_f64) * ((f).tanh())) + ((e).cosh()) + ((d).sinh())

}




#[inline]
pub  fn test_vector_6_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    ((-1_f64) * (((c) + (d)).signum())) + (((a) + ((-1_f64) * (b))).signum())

}




#[inline]
pub  fn test_vector_7_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    ((d).powf((e) + (f))) + (((((a) + (b)).powi(2)) * ((c).exp())).sqrt())

}




#[inline]
pub  fn test_vector_8_0(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {

    (((d).exp()).max((e).cos())) + (((c).sin()).min(((a) + (b)).ln()))

}


pub fn test_vector(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> [f64; 9] {

    let mut vec: [f64; 9] = [0_f64; 9];
    
vec[0] = test_vector_0_0(a, b, c, d, e, f);


vec[1] = test_vector_1_0(a, b, c, d, e, f);


vec[2] = test_vector_2_0(a, b, c, d, e, f);


vec[3] = test_vector_3_0(a, b, c, d, e, f);


vec[4] = test_vector_4_0(a, b, c, d, e, f);


vec[5] = test_vector_5_0(a, b, c, d, e, f);


vec[6] = test_vector_6_0(a, b, c, d, e, f);


vec[7] = test_vector_7_0(a, b, c, d, e, f);


vec[8] = test_vector_8_0(a, b, c, d, e, f);

    
    vec
}