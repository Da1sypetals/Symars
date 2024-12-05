use criterion::{criterion_group, criterion_main, Criterion};
use serde::{Deserialize, Serialize};
use std::fs;
use symars_demonstration::sym::gen_cached::{dphi_dx, dphi_dx_cached};

#[derive(Serialize, Deserialize)]
struct Args {
    args: Vec<f64>,
}

fn cached(c: &mut Criterion) {
    let args: Args = toml::from_str(&fs::read_to_string("fp.toml").unwrap()).unwrap();
    let args = args.args;
    c.bench_function("Cached", |b| {
        b.iter(|| {
            dphi_dx_cached(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8],
                args[9], args[10], args[11], args[12], args[13],
            )
        })
    });
}

fn uncached(c: &mut Criterion) {
    c.bench_function("Not cached", |b| {
        let args: Args = toml::from_str(&fs::read_to_string("fp.toml").unwrap()).unwrap();
        let args = args.args;
        b.iter(|| {
            dphi_dx(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8],
                args[9], args[10], args[11], args[12], args[13],
            )
        })
    });
}

criterion_group!(benches, cached, uncached);
criterion_main!(benches);
