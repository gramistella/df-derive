use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS: usize = 50_000;

#[derive(ToDataFrame, Clone)]
struct MixedTuple(i64, Option<f64>, Vec<bool>, String, u32);

#[derive(ToDataFrame, Clone)]
struct InnerTuple(i32, u32);

#[derive(ToDataFrame, Clone)]
struct OuterTuple(i32, InnerTuple);

fn generate_mixed_tuples() -> Vec<MixedTuple> {
    (0..N_ROWS)
        .map(|i| {
            let f = i64::try_from(i).unwrap();
            MixedTuple(
                1_700_000_000 + f,
                if i % 7 == 0 {
                    None
                } else {
                    Some(f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.0))
                },
                (0..=(i % 4)).map(|k| (k + i) % 2 == 0).collect(),
                format!("row-{i}"),
                u32::try_from(i).unwrap() % 1_000,
            )
        })
        .collect()
}

fn generate_outer_tuples() -> Vec<OuterTuple> {
    (0..N_ROWS)
        .map(|i| {
            let i32_i = i32::try_from(i).unwrap();
            OuterTuple(
                i32_i,
                InnerTuple(i32_i.wrapping_mul(3), u32::try_from(i).unwrap()),
            )
        })
        .collect()
}

fn benchmark_tuple_structs(c: &mut Criterion) {
    let mixed = generate_mixed_tuples();
    let nested = generate_outer_tuples();

    c.bench_function("tuple_structs_conversion", |b| {
        b.iter(|| {
            let df_mixed = std::hint::black_box(&mixed).to_dataframe().unwrap();
            let df_nested = std::hint::black_box(&nested).to_dataframe().unwrap();
            std::hint::black_box((df_mixed, df_nested))
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_tuple_structs
}
criterion_main!(benches);
