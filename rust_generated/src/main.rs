use symars_demonstration::sym::{
    gen_cached::{dphi_dx, dphi_dx_cached},
    gen_mat_nalgebra,
};

fn main() {
    let mat = gen_mat_nalgebra::test_matrix(1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    dbg!(mat);

    let cached = dphi_dx_cached(
        3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0,
    );
    let uncached = dphi_dx(
        3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0,
    );

    dbg!(cached);
    dbg!(uncached);
}
