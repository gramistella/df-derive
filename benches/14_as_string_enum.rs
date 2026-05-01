use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;

#[derive(Clone, Copy)]
enum Side {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
            Self::Hold => write!(f, "HOLD"),
        }
    }
}

#[derive(ToDataFrame, Clone)]
struct Trade {
    id: u64,
    price: f64,
    #[df_derive(as_string)]
    side: Side,
    #[df_derive(as_string)]
    side_opt: Option<Side>,
}

fn generate_trades() -> Vec<Trade> {
    (0..N_ROWS)
        .map(|i| {
            let side = match i % 3 {
                0 => Side::Buy,
                1 => Side::Sell,
                _ => Side::Hold,
            };
            Trade {
                id: i as u64,
                price: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.0),
                side,
                side_opt: if i % 5 == 0 { None } else { Some(side) },
            }
        })
        .collect()
}

fn benchmark_as_string_enum(c: &mut Criterion) {
    let data = generate_trades();

    c.bench_function("as_string_enum_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_as_string_enum
}
criterion_main!(benches);
