// Empirical validation of the smart-pointer transparency overhead. Each
// pair of structs (with vs. without smart pointers) runs through the
// columnar `to_dataframe` path on 100k rows. The macro's deref-rewriting
// should compile to equivalent MIR; any time delta reflects the underlying
// layout cost of indirection (heap-allocated `Box<u64>` vs. inline `u64`),
// not the macro overhead.
//
// `clippy::vec_box` and similar lints are suppressed because the
// pathological smart-pointer shapes are the entire point of the benchmark.

#![allow(clippy::vec_box)]
#![allow(clippy::redundant_allocation)]

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

use chrono::NaiveDate;
use std::sync::Arc;

const N_ROWS: usize = 100_000;

// --- Pair 1: bare scalar / Box / Arc ---

#[derive(ToDataFrame, Clone)]
struct U64Plain {
    v: Vec<u64>,
}

#[derive(ToDataFrame, Clone)]
struct U64Box {
    v: Vec<Box<u64>>,
}

#[derive(ToDataFrame, Clone)]
struct U64Arc {
    v: Vec<Arc<u64>>,
}

// --- Pair 2: bare String / Arc<String> ---

#[derive(ToDataFrame, Clone)]
struct StrPlain {
    s: Vec<String>,
}

#[derive(ToDataFrame, Clone)]
struct StrArc {
    s: Vec<Arc<String>>,
}

// --- Pair 3: bare NaiveDate / Box<NaiveDate> ---

#[derive(ToDataFrame, Clone)]
struct DatePlain {
    d: Vec<NaiveDate>,
}

#[derive(ToDataFrame, Clone)]
struct DateBox {
    d: Vec<Box<NaiveDate>>,
}

// --- Pair 4: Option<i32> / Option<Box<i32>> ---

#[derive(ToDataFrame, Clone)]
struct OptI32Plain {
    v: Option<i32>,
}

#[derive(ToDataFrame, Clone)]
struct OptI32Box {
    v: Option<Box<i32>>,
}

fn make_u64_plain() -> Vec<U64Plain> {
    (0..N_ROWS)
        .map(|i| U64Plain {
            v: vec![i as u64; 4],
        })
        .collect()
}
fn make_u64_box() -> Vec<U64Box> {
    (0..N_ROWS)
        .map(|i| U64Box {
            v: (0_u64..4).map(|j| Box::new(i as u64 + j)).collect(),
        })
        .collect()
}
fn make_u64_arc() -> Vec<U64Arc> {
    (0..N_ROWS)
        .map(|i| U64Arc {
            v: (0_u64..4).map(|j| Arc::new(i as u64 + j)).collect(),
        })
        .collect()
}
fn make_str_plain() -> Vec<StrPlain> {
    (0..N_ROWS)
        .map(|i| StrPlain {
            s: vec![format!("row-{i}-a"), format!("row-{i}-b")],
        })
        .collect()
}
fn make_str_arc() -> Vec<StrArc> {
    (0..N_ROWS)
        .map(|i| StrArc {
            s: vec![
                Arc::new(format!("row-{i}-a")),
                Arc::new(format!("row-{i}-b")),
            ],
        })
        .collect()
}
fn make_date_plain() -> Vec<DatePlain> {
    let epoch = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    (0..N_ROWS)
        .map(|i| DatePlain {
            d: vec![epoch + chrono::Duration::days(i64::try_from(i % 365).unwrap()); 3],
        })
        .collect()
}
fn make_date_box() -> Vec<DateBox> {
    let epoch = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    (0..N_ROWS)
        .map(|i| DateBox {
            d: (0..3)
                .map(|_| Box::new(epoch + chrono::Duration::days(i64::try_from(i % 365).unwrap())))
                .collect(),
        })
        .collect()
}
fn make_opt_i32_plain() -> Vec<OptI32Plain> {
    (0..N_ROWS)
        .map(|i| OptI32Plain {
            v: if i % 5 == 0 {
                None
            } else {
                Some(i32::try_from(i).unwrap())
            },
        })
        .collect()
}
fn make_opt_i32_box() -> Vec<OptI32Box> {
    (0..N_ROWS)
        .map(|i| OptI32Box {
            v: if i % 5 == 0 {
                None
            } else {
                Some(Box::new(i32::try_from(i).unwrap()))
            },
        })
        .collect()
}

fn bench_smart_pointers(c: &mut Criterion) {
    let u64_plain = make_u64_plain();
    let u64_box = make_u64_box();
    let u64_arc = make_u64_arc();
    let str_plain = make_str_plain();
    let str_arc = make_str_arc();
    let date_plain = make_date_plain();
    let date_box = make_date_box();
    let opt_i32_plain = make_opt_i32_plain();
    let opt_i32_box = make_opt_i32_box();

    let mut g = c.benchmark_group("smart_pointers");
    g.bench_function("vec_u64_plain", |b| {
        b.iter(|| std::hint::black_box(&u64_plain).to_dataframe().unwrap());
    });
    g.bench_function("vec_u64_box", |b| {
        b.iter(|| std::hint::black_box(&u64_box).to_dataframe().unwrap());
    });
    g.bench_function("vec_u64_arc", |b| {
        b.iter(|| std::hint::black_box(&u64_arc).to_dataframe().unwrap());
    });
    g.bench_function("vec_str_plain", |b| {
        b.iter(|| std::hint::black_box(&str_plain).to_dataframe().unwrap());
    });
    g.bench_function("vec_str_arc", |b| {
        b.iter(|| std::hint::black_box(&str_arc).to_dataframe().unwrap());
    });
    g.bench_function("vec_date_plain", |b| {
        b.iter(|| std::hint::black_box(&date_plain).to_dataframe().unwrap());
    });
    g.bench_function("vec_date_box", |b| {
        b.iter(|| std::hint::black_box(&date_box).to_dataframe().unwrap());
    });
    g.bench_function("opt_i32_plain", |b| {
        b.iter(|| std::hint::black_box(&opt_i32_plain).to_dataframe().unwrap());
    });
    g.bench_function("opt_i32_box", |b| {
        b.iter(|| std::hint::black_box(&opt_i32_box).to_dataframe().unwrap());
    });
    g.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_smart_pointers
}
criterion_main!(benches);
