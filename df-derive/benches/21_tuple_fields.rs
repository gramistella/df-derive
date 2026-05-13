// Empirical validation that tuple-field encoding stays close to the
// equivalent named-field shape. Tuple fields expand into per-projection
// column pipelines, so this bench keeps the tuple and named variants visible
// side by side.

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;

// --- Pair 1: bare tuple field vs. two named fields ---

#[derive(ToDataFrame, Clone)]
struct PairTuple {
    pair: (i32, String),
}

#[derive(ToDataFrame, Clone)]
struct PairNamed {
    pair_0: i32,
    pair_1: String,
}

// --- Pair 2: Vec<(i32, f64)> vs. two parallel Vecs ---

#[derive(ToDataFrame, Clone)]
struct VecTuple {
    pairs: Vec<(i32, f64)>,
}

#[derive(ToDataFrame, Clone)]
struct VecNamed {
    pairs_0: Vec<i32>,
    pairs_1: Vec<f64>,
}

// --- Pair 3: nested tuple ((i32, String), bool) vs. flat fields ---

#[derive(ToDataFrame, Clone)]
struct NestedTuple {
    nested: ((i32, String), bool),
}

#[derive(ToDataFrame, Clone)]
// Names mirror the nested tuple projection columns this benchmark compares.
#[allow(clippy::struct_field_names)]
struct NestedNamed {
    nested_0_0: i32,
    nested_0_1: String,
    nested_1: bool,
}

fn make_pair_tuple() -> Vec<PairTuple> {
    (0..N_ROWS)
        .map(|i| PairTuple {
            pair: (i32::try_from(i).unwrap(), format!("row-{i}")),
        })
        .collect()
}

fn make_pair_named() -> Vec<PairNamed> {
    (0..N_ROWS)
        .map(|i| PairNamed {
            pair_0: i32::try_from(i).unwrap(),
            pair_1: format!("row-{i}"),
        })
        .collect()
}

fn make_vec_tuple() -> Vec<VecTuple> {
    (0..N_ROWS)
        .map(|i| {
            let f = u32::try_from(i).unwrap();
            VecTuple {
                pairs: (0_u32..4)
                    .map(|j| (i32::try_from(j).unwrap(), f64::from(f + j) * 0.5))
                    .collect(),
            }
        })
        .collect()
}

fn make_vec_named() -> Vec<VecNamed> {
    (0..N_ROWS)
        .map(|i| {
            let f = u32::try_from(i).unwrap();
            VecNamed {
                pairs_0: (0..4i32).collect(),
                pairs_1: (0_u32..4).map(|j| f64::from(f + j) * 0.5).collect(),
            }
        })
        .collect()
}

fn make_nested_tuple() -> Vec<NestedTuple> {
    (0..N_ROWS)
        .map(|i| NestedTuple {
            nested: ((i32::try_from(i).unwrap(), format!("row-{i}")), i % 2 == 0),
        })
        .collect()
}

fn make_nested_named() -> Vec<NestedNamed> {
    (0..N_ROWS)
        .map(|i| NestedNamed {
            nested_0_0: i32::try_from(i).unwrap(),
            nested_0_1: format!("row-{i}"),
            nested_1: i % 2 == 0,
        })
        .collect()
}

fn bench_tuple_fields(c: &mut Criterion) {
    let pair_tuple = make_pair_tuple();
    let pair_named = make_pair_named();
    let vec_tuple = make_vec_tuple();
    let vec_named = make_vec_named();
    let nested_tuple = make_nested_tuple();
    let nested_named = make_nested_named();

    let mut g = c.benchmark_group("tuple_fields");
    g.bench_function("pair_tuple", |b| {
        b.iter(|| std::hint::black_box(&pair_tuple).to_dataframe().unwrap());
    });
    g.bench_function("pair_named", |b| {
        b.iter(|| std::hint::black_box(&pair_named).to_dataframe().unwrap());
    });
    g.bench_function("vec_tuple", |b| {
        b.iter(|| std::hint::black_box(&vec_tuple).to_dataframe().unwrap());
    });
    g.bench_function("vec_named", |b| {
        b.iter(|| std::hint::black_box(&vec_named).to_dataframe().unwrap());
    });
    g.bench_function("nested_tuple", |b| {
        b.iter(|| std::hint::black_box(&nested_tuple).to_dataframe().unwrap());
    });
    g.bench_function("nested_named", |b| {
        b.iter(|| std::hint::black_box(&nested_named).to_dataframe().unwrap());
    });
    g.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_tuple_fields
}
criterion_main!(benches);
