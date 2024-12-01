use symars_demonstration::sym::gen_mat_nalgebra;

fn main() {
    let mat = gen_mat_nalgebra::test_matrix(1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    dbg!(mat);
}
