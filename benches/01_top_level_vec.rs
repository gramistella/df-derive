use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;
use df_derive::ToDataFrame;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct Tick {
    ts: i64,
    price: f64,
    volume: u64,
    bid: f64,
    ask: f64,
    bid_size: u32,
}

fn generate_ticks() -> Vec<Tick> {
    (0..N_ROWS)
        .map(|i| Tick {
            ts: 1_700_000_000 + i64::try_from(i).unwrap(),
            price: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.0),
            volume: 1_000 + (i as u64),
            bid: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 99.9),
            ask: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.1),
            bid_size: 10 + (u32::try_from(i).unwrap() % 100),
        })
        .collect()
}

fn benchmark_top_level_vec(c: &mut Criterion) {
    let data = generate_ticks();

    c.bench_function("top_level_vec_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
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
    targets = benchmark_top_level_vec
}
criterion_main!(benches);
