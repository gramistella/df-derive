//! Generic-field conversions: a generic struct instantiated with a unit, a
//! primitive and a nested struct payload, plus `Option<T>` and `Vec<T>` around
//! a nested payload.

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
use crate::bench_support::configure_criterion;
use df_derive::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct Meta {
    timestamp: i64,
    note: String,
}

#[derive(ToDataFrame, Clone)]
struct Wrapper<T>
where
    T: Clone,
{
    id: u32,
    price: f64,
    payload: T,
}

#[derive(ToDataFrame, Clone)]
struct OptWrap<T>
where
    T: Clone,
{
    id: u32,
    payload: Option<T>,
}

#[derive(ToDataFrame, Clone)]
struct VecWrap<T>
where
    T: Clone,
{
    id: u32,
    payload: Vec<T>,
}

fn generate_with_unit() -> Vec<Wrapper<()>> {
    (0..N_ROWS)
        .map(|i| Wrapper {
            id: u32::try_from(i).unwrap(),
            price: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.0),
            payload: (),
        })
        .collect()
}

fn generate_with_primitive() -> Vec<Wrapper<f64>> {
    (0..N_ROWS)
        .map(|i| Wrapper {
            id: u32::try_from(i).unwrap(),
            price: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.0),
            payload: f64::from(u32::try_from(i).unwrap()),
        })
        .collect()
}

fn generate_with_struct() -> Vec<Wrapper<Meta>> {
    (0..N_ROWS)
        .map(|i| Wrapper {
            id: u32::try_from(i).unwrap(),
            price: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.0),
            payload: Meta {
                timestamp: 1_700_000_000 + i64::try_from(i).unwrap(),
                note: "n".to_string(),
            },
        })
        .collect()
}

fn generate_opt_struct() -> Vec<OptWrap<Meta>> {
    (0..N_ROWS)
        .map(|i| OptWrap {
            id: u32::try_from(i).unwrap(),
            payload: if i % 3 == 0 {
                None
            } else {
                Some(Meta {
                    timestamp: 1_700_000_000 + i64::try_from(i).unwrap(),
                    note: "n".to_string(),
                })
            },
        })
        .collect()
}

fn generate_vec_struct() -> Vec<VecWrap<Meta>> {
    (0..N_ROWS / 100)
        .map(|i| {
            let len = (i % 5) + 1;
            VecWrap {
                id: u32::try_from(i).unwrap(),
                payload: (0..len)
                    .map(|j| Meta {
                        timestamp: 1_700_000_000 + i64::try_from(i * 10 + j).unwrap(),
                        note: "n".to_string(),
                    })
                    .collect(),
            }
        })
        .collect()
}

fn benchmark_generics(c: &mut Criterion) {
    let unit_data = generate_with_unit();
    c.bench_function("generics_unit_bulk", |b| {
        b.iter(|| std::hint::black_box(std::hint::black_box(&unit_data).to_dataframe().unwrap()));
    });

    let prim_data = generate_with_primitive();
    c.bench_function("generics_primitive_bulk", |b| {
        b.iter(|| std::hint::black_box(std::hint::black_box(&prim_data).to_dataframe().unwrap()));
    });

    let struct_data = generate_with_struct();
    c.bench_function("generics_struct_bulk", |b| {
        b.iter(|| std::hint::black_box(std::hint::black_box(&struct_data).to_dataframe().unwrap()));
    });

    let opt_data = generate_opt_struct();
    c.bench_function("opt_struct_bulk", |b| {
        b.iter(|| std::hint::black_box(std::hint::black_box(&opt_data).to_dataframe().unwrap()));
    });

    let vec_data = generate_vec_struct();
    c.bench_function("vec_struct_bulk", |b| {
        b.iter(|| std::hint::black_box(std::hint::black_box(&vec_data).to_dataframe().unwrap()));
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_generics
}
criterion_main!(benches);
