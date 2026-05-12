use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

use chrono::{DateTime, TimeZone, Utc};
use rust_decimal::Decimal;

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct Tick {
    #[df_derive(time_unit = "ms")]
    ts: DateTime<Utc>,
    #[df_derive(time_unit = "ms")]
    ts_opt: Option<DateTime<Utc>>,
    #[df_derive(decimal(precision = 18, scale = 6))]
    price: Decimal,
    #[df_derive(decimal(precision = 18, scale = 6))]
    price_opt: Option<Decimal>,
    volume: u64,
}

fn generate_ticks() -> Vec<Tick> {
    let base = Utc
        .timestamp_millis_opt(1_700_000_000_000)
        .single()
        .unwrap();
    (0..N_ROWS)
        .map(|i| {
            let i_i64 = i64::try_from(i).unwrap();
            let ts = base + chrono::Duration::milliseconds(i_i64 * 100);
            Tick {
                ts,
                ts_opt: if i % 7 == 0 { None } else { Some(ts) },
                price: Decimal::new(1_000_000 + i_i64 * 13, 6),
                price_opt: if i % 5 == 0 {
                    None
                } else {
                    Some(Decimal::new(2_000_000 + i_i64 * 11, 6))
                },
                volume: 1_000 + (i as u64),
            }
        })
        .collect()
}

fn benchmark_datetime_decimal_columns(c: &mut Criterion) {
    let data = generate_ticks();

    c.bench_function("datetime_decimal_columns_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_datetime_decimal_columns
}
criterion_main!(benches);
