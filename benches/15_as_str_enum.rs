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
enum SideStr {
    Buy,
    Sell,
    Hold,
}

impl AsRef<str> for SideStr {
    fn as_ref(&self) -> &str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
            Self::Hold => "HOLD",
        }
    }
}

#[derive(ToDataFrame, Clone)]
struct Trade {
    id: u64,
    price: f64,
    #[df_derive(as_str)]
    side: SideStr,
    #[df_derive(as_str)]
    side_opt: Option<SideStr>,
}

fn generate_trades() -> Vec<Trade> {
    (0..N_ROWS)
        .map(|i| {
            let side = match i % 3 {
                0 => SideStr::Buy,
                1 => SideStr::Sell,
                _ => SideStr::Hold,
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

fn benchmark_as_str_enum(c: &mut Criterion) {
    let data = generate_trades();

    c.bench_function("as_str_enum_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_as_str_enum
}
criterion_main!(benches);
