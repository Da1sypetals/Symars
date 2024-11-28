fn main() {
    let a: f64 = 1.0;
    let b: f64 = -1.0;
    let c: f64 = 1.0;
    let d: f64 = 2.0;
    let e: f64 = 1.0;
    let f: f64 = 2.0;
    let g: f64 = 1.0;
    let h: f64 = 2.0;
    let x: f64 = 2.0;
    let y: f64 = 2.0;

    let z = {
        //
        ((-1_f64) * (((a) + (h)).exp()))
            + ((d) * ((e).recip()) * ((a) + (b) + ((-1_f64) * (c))))
            + ((((-0.2_f64) * (h)) + ((f) * (g))).ln())
        //
    };

    dbg!(z);
}
