use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use std::time::Duration;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrameVec;

use chrono::{DateTime, TimeZone, Utc};
use rust_decimal::Decimal;

const N_ROWS: usize = 10_000;

#[derive(ToDataFrame, Clone)]
struct VecOptString {
    id: u64,
    items: Vec<Option<String>>,
}

#[derive(ToDataFrame, Clone)]
struct VecOptDateTime {
    id: u64,
    #[df_derive(time_unit = "ms")]
    items: Vec<Option<DateTime<Utc>>>,
}

#[derive(ToDataFrame, Clone)]
struct VecOptDecimal {
    id: u64,
    #[df_derive(decimal(precision = 18, scale = 6))]
    items: Vec<Option<Decimal>>,
}

#[derive(ToDataFrame, Clone)]
struct VecVecI32 {
    id: u64,
    items: Vec<Vec<i32>>,
}

fn generate_vec_opt_string() -> Vec<VecOptString> {
    (0..N_ROWS)
        .map(|i| VecOptString {
            id: i as u64,
            items: (0..(i % 7 + 3))
                .map(|k| {
                    if (k + i) % 4 == 0 {
                        None
                    } else {
                        Some(format!("val-{i}-{k}"))
                    }
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_opt_datetime() -> Vec<VecOptDateTime> {
    let base = Utc
        .timestamp_millis_opt(1_700_000_000_000)
        .single()
        .unwrap();
    (0..N_ROWS)
        .map(|i| VecOptDateTime {
            id: i as u64,
            items: (0..(i % 7 + 3))
                .map(|k| {
                    if (k + i) % 4 == 0 {
                        None
                    } else {
                        Some(base + chrono::Duration::seconds(i64::try_from(i * 100 + k).unwrap()))
                    }
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_opt_decimal() -> Vec<VecOptDecimal> {
    (0..N_ROWS)
        .map(|i| VecOptDecimal {
            id: i as u64,
            items: (0..(i % 7 + 3))
                .map(|k| {
                    if (k + i) % 4 == 0 {
                        None
                    } else {
                        Some(Decimal::new(i64::try_from(i * 1000 + k).unwrap(), 4))
                    }
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_vec_i32() -> Vec<VecVecI32> {
    (0..N_ROWS)
        .map(|i| VecVecI32 {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| {
                    (0..(j % 4 + 2))
                        .map(|k| {
                            i32::try_from(i).unwrap() * 10
                                + i32::try_from(j).unwrap()
                                + i32::try_from(k).unwrap()
                        })
                        .collect()
                })
                .collect(),
        })
        .collect()
}

fn benchmark_vec_opt_string(c: &mut Criterion) {
    let data = generate_vec_opt_string();
    c.bench_function("vec_opt_string", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn benchmark_vec_opt_datetime(c: &mut Criterion) {
    let data = generate_vec_opt_datetime();
    c.bench_function("vec_opt_datetime", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn benchmark_vec_opt_decimal(c: &mut Criterion) {
    let data = generate_vec_opt_decimal();
    c.bench_function("vec_opt_decimal", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn benchmark_vec_vec_i32(c: &mut Criterion) {
    let data = generate_vec_vec_i32();
    c.bench_function("vec_vec_i32", |b| {
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
    targets =
        benchmark_vec_opt_string,
        benchmark_vec_opt_datetime,
        benchmark_vec_opt_decimal,
        benchmark_vec_vec_i32
}
criterion_main!(benches);
