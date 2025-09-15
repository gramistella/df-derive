use criterion::{Criterion, black_box, criterion_group, criterion_main};
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
            ts: 1_700_000_000 + i as i64,
            price: 100.0 + (i as f64) * 0.001,
            volume: 1_000 + (i as u64),
            bid: 99.9 + (i as f64) * 0.001,
            ask: 100.1 + (i as f64) * 0.001,
            bid_size: 10 + (i as u32 % 100),
        })
        .collect()
}

fn benchmark_top_level_vec(c: &mut Criterion) {
    let data = generate_ticks();

    c.bench_function("top_level_vec_conversion", |b| {
        b.iter(|| {
            let df = black_box(&data).to_dataframe().unwrap();
            black_box(df)
        })
    });
}

criterion_group!(benches, benchmark_top_level_vec);
criterion_main!(benches);
