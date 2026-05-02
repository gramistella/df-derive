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

#[derive(ToDataFrame, Clone)]
struct VecVecOptI32 {
    id: u64,
    items: Vec<Vec<Option<i32>>>,
}

#[derive(ToDataFrame, Clone)]
struct VecVecString {
    id: u64,
    items: Vec<Vec<String>>,
}

#[derive(ToDataFrame, Clone)]
struct VecOptBool {
    id: u64,
    items: Vec<Option<bool>>,
}

#[derive(ToDataFrame, Clone)]
struct VecVecBool {
    id: u64,
    items: Vec<Vec<bool>>,
}

#[derive(ToDataFrame, Clone)]
struct VecVecOptString {
    id: u64,
    items: Vec<Vec<Option<String>>>,
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

fn generate_vec_vec_opt_i32() -> Vec<VecVecOptI32> {
    (0..N_ROWS)
        .map(|i| VecVecOptI32 {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| {
                    (0..(j % 4 + 2))
                        .map(|k| {
                            if (i + j + k) % 5 == 0 {
                                None
                            } else {
                                Some(
                                    i32::try_from(i).unwrap() * 10
                                        + i32::try_from(j).unwrap()
                                        + i32::try_from(k).unwrap(),
                                )
                            }
                        })
                        .collect()
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_vec_string() -> Vec<VecVecString> {
    (0..N_ROWS)
        .map(|i| VecVecString {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| (0..(j % 4 + 2)).map(|k| format!("s-{i}-{j}-{k}")).collect())
                .collect(),
        })
        .collect()
}

fn generate_vec_opt_bool() -> Vec<VecOptBool> {
    (0..N_ROWS)
        .map(|i| VecOptBool {
            id: i as u64,
            items: (0..(i % 7 + 3))
                .map(|k| {
                    if (k + i) % 5 == 0 {
                        None
                    } else {
                        Some((k + i) % 2 == 0)
                    }
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_vec_bool() -> Vec<VecVecBool> {
    (0..N_ROWS)
        .map(|i| VecVecBool {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| (0..(j % 4 + 2)).map(|k| (i + j + k) % 2 == 0).collect())
                .collect(),
        })
        .collect()
}

fn generate_vec_vec_opt_string() -> Vec<VecVecOptString> {
    (0..N_ROWS)
        .map(|i| VecVecOptString {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| {
                    (0..(j % 4 + 2))
                        .map(|k| {
                            if (i + j + k) % 4 == 0 {
                                None
                            } else {
                                Some(format!("s-{i}-{j}-{k}"))
                            }
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

fn benchmark_vec_vec_opt_i32(c: &mut Criterion) {
    let data = generate_vec_vec_opt_i32();
    c.bench_function("vec_vec_opt_i32", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn benchmark_vec_vec_string(c: &mut Criterion) {
    let data = generate_vec_vec_string();
    c.bench_function("vec_vec_string", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn benchmark_vec_opt_bool(c: &mut Criterion) {
    let data = generate_vec_opt_bool();
    c.bench_function("vec_opt_bool", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn benchmark_vec_vec_bool(c: &mut Criterion) {
    let data = generate_vec_vec_bool();
    c.bench_function("vec_vec_bool", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn benchmark_vec_vec_opt_string(c: &mut Criterion) {
    let data = generate_vec_vec_opt_string();
    c.bench_function("vec_vec_opt_string", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        benchmark_vec_opt_string,
        benchmark_vec_opt_datetime,
        benchmark_vec_opt_decimal,
        benchmark_vec_vec_i32,
        benchmark_vec_vec_opt_i32,
        benchmark_vec_vec_string,
        benchmark_vec_opt_bool,
        benchmark_vec_vec_bool,
        benchmark_vec_vec_opt_string
}
criterion_main!(benches);
