use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use std::time::Duration;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

const NUM_QUOTES: usize = 100_000;

#[derive(ToDataFrame)]
struct Quote {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

#[derive(ToDataFrame)]
struct MarketData {
    symbol: String,
    quotes: Vec<Quote>,
}

fn generate_market_data() -> MarketData {
    let quotes: Vec<Quote> = (0..NUM_QUOTES)
        .map(|i| {
            let i_f64 = f64::from(u32::try_from(i).unwrap());
            Quote {
                timestamp: 1_700_000_000 + i64::try_from(i).unwrap(),
                open: i_f64.mul_add(0.1, 100.0),
                high: i_f64.mul_add(0.1, 102.0),
                low: i_f64.mul_add(0.1, 99.5),
                close: i_f64.mul_add(0.1, 101.0),
                volume: 1000 + i as u64,
            }
        })
        .collect();

    MarketData {
        symbol: "BENCH".to_string(),
        quotes,
    }
}

fn benchmark_vec_custom_struct(c: &mut Criterion) {
    let market_data = generate_market_data();

    c.bench_function("vec_custom_struct_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&market_data).to_dataframe().unwrap();
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
    targets = benchmark_vec_custom_struct
}
criterion_main!(benches);
