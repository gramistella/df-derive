use criterion::{Criterion, black_box, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

const N_HOSTS: usize = 10_000;

#[derive(ToDataFrame, Clone)]
struct Metric {
    name: String,
    value: f64,
}

#[derive(ToDataFrame, Clone)]
struct Host {
    id: u64,
    metrics: Vec<Metric>,
}

fn generate_hosts() -> Vec<Host> {
    // All hosts have empty metrics vectors; ensures List columns get empty inner series per row
    (0..N_HOSTS)
        .map(|i| Host {
            id: i as u64,
            metrics: Vec::new(),
        })
        .collect()
}

fn benchmark_empty_nested_vec(c: &mut Criterion) {
    let hosts = generate_hosts();

    c.bench_function("empty_nested_vec_conversion", |b| {
        b.iter(|| {
            let df = black_box(&hosts).to_dataframe().unwrap();
            black_box(df)
        })
    });
}

criterion_group!(benches, benchmark_empty_nested_vec);
criterion_main!(benches);
