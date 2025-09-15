use criterion::{Criterion, black_box, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS: usize = 50_000;

#[derive(ToDataFrame, Clone)]
struct WideTick {
    ts: i64,
    price: Option<f64>,
    volume: Option<u64>,
    bid: Option<f64>,
    ask: Option<f64>,
    bid_size: Option<u32>,
    ask_size: Option<u32>,
    trade_id: Option<String>,
    venue: Option<String>,
    flag_a: Option<bool>,
    flag_b: Option<bool>,
}

fn generate_wide_ticks() -> Vec<WideTick> {
    (0..N_ROWS)
        .map(|i| WideTick {
            ts: 1_700_000_000 + i as i64,
            price: if i % 10 == 0 {
                None
            } else {
                Some(100.0 + i as f64 * 0.01)
            },
            volume: if i % 13 == 0 {
                None
            } else {
                Some(1_000 + i as u64)
            },
            bid: if i % 7 == 0 {
                None
            } else {
                Some(99.5 + i as f64 * 0.01)
            },
            ask: if i % 11 == 0 {
                None
            } else {
                Some(100.5 + i as f64 * 0.01)
            },
            bid_size: if i % 9 == 0 {
                None
            } else {
                Some(10 + (i as u32 % 100))
            },
            ask_size: if i % 8 == 0 {
                None
            } else {
                Some(10 + (i as u32 % 100))
            },
            trade_id: if i % 6 == 0 {
                None
            } else {
                Some(format!("T-{i:08}"))
            },
            venue: if i % 5 == 0 {
                None
            } else {
                Some("XNYS".to_string())
            },
            flag_a: if i % 4 == 0 { None } else { Some(i % 2 == 0) },
            flag_b: if i % 3 == 0 { None } else { Some(i % 2 == 1) },
        })
        .collect()
}

fn benchmark_wide_top_level_options(c: &mut Criterion) {
    let data = generate_wide_ticks();

    c.bench_function("wide_top_level_options_conversion", |b| {
        b.iter(|| {
            let df = black_box(&data).to_dataframe().unwrap();
            black_box(df)
        })
    });
}

criterion_group!(benches, benchmark_wide_top_level_options);
criterion_main!(benches);
