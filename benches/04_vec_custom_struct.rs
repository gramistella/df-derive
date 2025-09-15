use criterion::{Criterion, black_box, criterion_group, criterion_main};
use df_derive::ToDataFrame;

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
            let i_f64 = i as f64;
            Quote {
                timestamp: 1700000000 + i as i64,
                open: 100.0 + i_f64 * 0.1,
                high: 102.0 + i_f64 * 0.1,
                low: 99.5 + i_f64 * 0.1,
                close: 101.0 + i_f64 * 0.1,
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
            let df = black_box(&market_data).to_dataframe().unwrap();
            black_box(df)
        })
    });
}

criterion_group!(benches, benchmark_vec_custom_struct);
criterion_main!(benches);
