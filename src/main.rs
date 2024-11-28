use faer::MatMut;

fn modify(mut mat: MatMut<f64>) {
    mat[(1, 3)] = 2.44
}

fn main() {
    let mut a: faer::Mat<f64> = faer::Mat::zeros(3, 5);
    modify(a.as_mut());

    dbg!(a);
}
