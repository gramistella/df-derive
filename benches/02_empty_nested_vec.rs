use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use std::time::Duration;

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
            let df = std::hint::black_box(&hosts).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}
fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(150)
        .warm_up_time(Duration::from_secs(8))
        .measurement_time(Duration::from_secs(20))
        .nresamples(200_000)
        .noise_threshold(0.02)
        .confidence_level(0.99)
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_empty_nested_vec
}
criterion_main!(benches);
