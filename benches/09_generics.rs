use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::time::Duration;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::{Columnar, ToDataFrame, ToDataFrameVec};

const N_ROWS: usize = 100_000;

// A small payload struct used to exercise the nested-struct generic instantiation.
#[derive(ToDataFrame, Clone)]
struct Meta {
    timestamp: i64,
    note: String,
}

// A generic wrapper struct: this is what the macro must compile.
#[derive(ToDataFrame, Clone)]
struct Wrapper<T>
where
    T: Clone,
{
    id: u32,
    price: f64,
    payload: T,
}

// Local trait impls so Wrapper<f64> can flatten via a single column.
impl ToDataFrame for f64 {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        DataFrame::new(vec![Series::new("value".into(), &[*self]).into()])
    }
    fn empty_dataframe() -> PolarsResult<DataFrame> {
        DataFrame::new(vec![
            Series::new_empty("value".into(), &DataType::Float64).into(),
        ])
    }
    fn schema() -> PolarsResult<Vec<(&'static str, DataType)>> {
        Ok(vec![("value", DataType::Float64)])
    }
}

impl Columnar for f64 {
    fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
        DataFrame::new(vec![Series::new("value".into(), items).into()])
    }
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

fn benchmark_generics(c: &mut Criterion) {
    let unit_data = generate_with_unit();
    let prim_data = generate_with_primitive();
    let struct_data = generate_with_struct();

    c.bench_function("generics_wrapper_unit", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&unit_data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    c.bench_function("generics_wrapper_primitive", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&prim_data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    c.bench_function("generics_wrapper_struct", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&struct_data).to_dataframe().unwrap();
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
    targets = benchmark_generics
}
criterion_main!(benches);
