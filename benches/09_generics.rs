//! A/B benchmark for the generic-field columnar path.
//!
//! Two implementations are compared on the same `Wrapper<T>` shape:
//! - `_per_row`: hand-rolled mirror of the original codegen, where the generic
//!   field is decoded by calling `payload.to_dataframe()` once per item and
//!   extracting `AnyValues` into per-column accumulators.
//! - `_bulk`: the macro-generated path that collects `Vec<T>` once and calls
//!   `T::columnar_to_dataframe(&slice)` exactly once, then prefix-renames the
//!   resulting columns onto the parent `DataFrame`.

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::time::Duration;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::{Columnar, ToDataFrame};

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct Meta {
    timestamp: i64,
    note: String,
}

// The generic struct under test. The macro emits the bulk path for `payload`.
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
        DataFrame::new_infer_height(vec![Series::new("value".into(), &[*self]).into()])
    }
    fn empty_dataframe() -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![
            Series::new_empty("value".into(), &DataType::Float64).into(),
        ])
    }
    fn schema() -> PolarsResult<Vec<(&'static str, DataType)>> {
        Ok(vec![("value", DataType::Float64)])
    }
}

impl Columnar for f64 {
    fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![Series::new("value".into(), items).into()])
    }
}

// Hand-rolled per-row equivalent of `__df_derive_vec_to_inner_list_values`
// for `Wrapper<T>`. Mirrors what the macro emits for the per-row generic-leaf
// path (one tmp DataFrame per item, AnyValues accumulated per column). Used as
// the A/B baseline against the new bulk override.
fn wrapper_vec_to_inner_list_values_per_row<T>(
    items: &[Wrapper<T>],
) -> PolarsResult<Vec<AnyValue<'static>>>
where
    T: Clone + ToDataFrame + Columnar,
{
    if items.is_empty() {
        let mut out = Vec::new();
        for (_, dtype) in <Wrapper<T> as ToDataFrame>::schema()? {
            let inner = Series::new_empty("".into(), &dtype);
            out.push(AnyValue::List(inner));
        }
        return Ok(out);
    }

    let mut id_buf: Vec<u32> = Vec::with_capacity(items.len());
    let mut price_buf: Vec<f64> = Vec::with_capacity(items.len());
    let payload_schema = <T as ToDataFrame>::schema()?;
    let mut payload_cols: Vec<Vec<AnyValue<'static>>> = payload_schema
        .iter()
        .map(|_| Vec::with_capacity(items.len()))
        .collect();

    for it in items {
        id_buf.push(it.id);
        price_buf.push(it.price);
        let tmp = it.payload.to_dataframe()?;
        let names: Vec<String> = tmp
            .get_column_names()
            .iter()
            .map(ToString::to_string)
            .collect();
        for (j, name) in names.iter().enumerate() {
            payload_cols[j].push(tmp.column(name.as_str())?.get(0)?.into_static());
        }
    }

    let mut out = Vec::with_capacity(2 + payload_schema.len());
    out.push(AnyValue::List(Series::new("".into(), &id_buf)));
    out.push(AnyValue::List(Series::new("".into(), &price_buf)));
    for col in payload_cols {
        out.push(AnyValue::List(Series::new("".into(), &col)));
    }
    Ok(out)
}

// Hand-rolled per-row implementation that mirrors the previous generic-leaf
// codegen: for every item we call `payload.to_dataframe()` and accumulate
// AnyValues per column. This is what the macro used to do before the bulk
// rewrite — it builds N tiny DataFrames per call.
fn wrapper_per_row<T>(items: &[Wrapper<T>]) -> PolarsResult<DataFrame>
where
    T: Clone + ToDataFrame + Columnar,
{
    if items.is_empty() {
        return <Wrapper<T> as ToDataFrame>::empty_dataframe();
    }

    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    let mut prices: Vec<f64> = Vec::with_capacity(items.len());

    let payload_schema = <T as ToDataFrame>::schema()?;
    let mut payload_cols: Vec<Vec<AnyValue<'static>>> = payload_schema
        .iter()
        .map(|_| Vec::with_capacity(items.len()))
        .collect();

    for it in items {
        ids.push(it.id);
        prices.push(it.price);
        let tmp_df = it.payload.to_dataframe()?;
        let names: Vec<String> = tmp_df
            .get_column_names()
            .iter()
            .map(ToString::to_string)
            .collect();
        for (j, name) in names.iter().enumerate() {
            let v = tmp_df.column(name.as_str())?.get(0)?.into_static();
            payload_cols[j].push(v);
        }
    }

    let mut columns: Vec<Column> = Vec::with_capacity(2 + payload_schema.len());
    columns.push(Series::new("id".into(), &ids).into());
    columns.push(Series::new("price".into(), &prices).into());
    for (j, (col_name, _dtype)) in payload_schema.iter().enumerate() {
        let prefixed = format!("payload.{col_name}");
        let s = Series::new(prefixed.as_str().into(), &payload_cols[j]);
        columns.push(s.into());
    }

    DataFrame::new_infer_height(columns)
}

// Wrappers used to A/B benchmark the new Option<T> and Vec<T> bulk overrides.
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

// Hand-rolled per-row equivalent of `OptWrap<T>::columnar_to_dataframe` —
// mirrors the previous generic-leaf codegen that built one tmp DataFrame per
// item and pushed an `AnyValue::Null` for every `None`.
fn opt_wrap_per_row<T>(items: &[OptWrap<T>]) -> PolarsResult<DataFrame>
where
    T: Clone + ToDataFrame + Columnar,
{
    if items.is_empty() {
        return <OptWrap<T> as ToDataFrame>::empty_dataframe();
    }
    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    let payload_schema = <T as ToDataFrame>::schema()?;
    let mut payload_cols: Vec<Vec<AnyValue<'static>>> = payload_schema
        .iter()
        .map(|_| Vec::with_capacity(items.len()))
        .collect();
    for it in items {
        ids.push(it.id);
        match &it.payload {
            Some(v) => {
                let tmp = v.to_dataframe()?;
                let names: Vec<String> = tmp
                    .get_column_names()
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                for (j, name) in names.iter().enumerate() {
                    payload_cols[j].push(tmp.column(name.as_str())?.get(0)?.into_static());
                }
            }
            None => {
                for col in &mut payload_cols {
                    col.push(AnyValue::Null);
                }
            }
        }
    }
    let mut columns: Vec<Column> = Vec::with_capacity(1 + payload_schema.len());
    columns.push(Series::new("id".into(), &ids).into());
    for (j, (col_name, _dtype)) in payload_schema.iter().enumerate() {
        let prefixed = format!("payload.{col_name}");
        columns.push(Series::new(prefixed.as_str().into(), &payload_cols[j]).into());
    }
    DataFrame::new_infer_height(columns)
}

// Hand-rolled per-row equivalent of `VecWrap<T>::columnar_to_dataframe` —
// mirrors the previous generic-vec codegen that called `to_dataframe` on
// every element of every parent row's `Vec<T>` and stitched per-row inner
// lists together.
fn vec_wrap_per_row<T>(items: &[VecWrap<T>]) -> PolarsResult<DataFrame>
where
    T: Clone + ToDataFrame + Columnar,
{
    if items.is_empty() {
        return <VecWrap<T> as ToDataFrame>::empty_dataframe();
    }
    let mut ids: Vec<u32> = Vec::with_capacity(items.len());
    let payload_schema = <T as ToDataFrame>::schema()?;
    let mut payload_rows: Vec<Vec<AnyValue<'static>>> = payload_schema
        .iter()
        .map(|_| Vec::with_capacity(items.len()))
        .collect();
    for it in items {
        ids.push(it.id);
        let mut row_per_col: Vec<Vec<AnyValue<'static>>> = payload_schema
            .iter()
            .map(|_| Vec::with_capacity(it.payload.len()))
            .collect();
        for elem in &it.payload {
            let tmp = elem.to_dataframe()?;
            let names: Vec<String> = tmp
                .get_column_names()
                .iter()
                .map(ToString::to_string)
                .collect();
            for (j, name) in names.iter().enumerate() {
                row_per_col[j].push(tmp.column(name.as_str())?.get(0)?.into_static());
            }
        }
        for (j, vals) in row_per_col.into_iter().enumerate() {
            let inner = Series::new("".into(), &vals);
            payload_rows[j].push(AnyValue::List(inner));
        }
    }
    let mut columns: Vec<Column> = Vec::with_capacity(1 + payload_schema.len());
    columns.push(Series::new("id".into(), &ids).into());
    for (j, (col_name, _dtype)) in payload_schema.iter().enumerate() {
        let prefixed = format!("payload.{col_name}");
        columns.push(Series::new(prefixed.as_str().into(), &payload_rows[j]).into());
    }
    DataFrame::new_infer_height(columns)
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

#[allow(clippy::too_many_lines)]
fn benchmark_generics(c: &mut Criterion) {
    let unit_data = generate_with_unit();
    let prim_data = generate_with_primitive();
    let struct_data = generate_with_struct();

    // Unit payload
    c.bench_function("generics_unit_per_row", |b| {
        b.iter(|| {
            let df = wrapper_per_row(std::hint::black_box(&unit_data)).unwrap();
            std::hint::black_box(df)
        });
    });
    c.bench_function("generics_unit_bulk", |b| {
        b.iter(|| {
            let df =
                <Wrapper<()> as Columnar>::columnar_to_dataframe(std::hint::black_box(&unit_data))
                    .unwrap();
            std::hint::black_box(df)
        });
    });

    // Primitive payload
    c.bench_function("generics_primitive_per_row", |b| {
        b.iter(|| {
            let df = wrapper_per_row(std::hint::black_box(&prim_data)).unwrap();
            std::hint::black_box(df)
        });
    });
    c.bench_function("generics_primitive_bulk", |b| {
        b.iter(|| {
            let df =
                <Wrapper<f64> as Columnar>::columnar_to_dataframe(std::hint::black_box(&prim_data))
                    .unwrap();
            std::hint::black_box(df)
        });
    });

    // Struct payload
    c.bench_function("generics_struct_per_row", |b| {
        b.iter(|| {
            let df = wrapper_per_row(std::hint::black_box(&struct_data)).unwrap();
            std::hint::black_box(df)
        });
    });
    c.bench_function("generics_struct_bulk", |b| {
        b.iter(|| {
            let df = <Wrapper<Meta> as Columnar>::columnar_to_dataframe(std::hint::black_box(
                &struct_data,
            ))
            .unwrap();
            std::hint::black_box(df)
        });
    });

    // Helpers' vec-anyvalues path (used by outer structs that hold
    // `Vec<Wrapper<T>>`). Compares the new bulk override against a hand-rolled
    // per-row equivalent.
    c.bench_function("helpers_struct_per_row", |b| {
        b.iter(|| {
            let v = wrapper_vec_to_inner_list_values_per_row(std::hint::black_box(&struct_data))
                .unwrap();
            std::hint::black_box(v)
        });
    });
    c.bench_function("helpers_struct_bulk", |b| {
        b.iter(|| {
            let v = Wrapper::<Meta>::__df_derive_vec_to_inner_list_values(std::hint::black_box(
                &struct_data,
            ))
            .unwrap();
            std::hint::black_box(v)
        });
    });

    c.bench_function("helpers_primitive_per_row", |b| {
        b.iter(|| {
            let v =
                wrapper_vec_to_inner_list_values_per_row(std::hint::black_box(&prim_data)).unwrap();
            std::hint::black_box(v)
        });
    });
    c.bench_function("helpers_primitive_bulk", |b| {
        b.iter(|| {
            let v = Wrapper::<f64>::__df_derive_vec_to_inner_list_values(std::hint::black_box(
                &prim_data,
            ))
            .unwrap();
            std::hint::black_box(v)
        });
    });

    // Option<T> direct: 100k rows, ~33% None.
    let opt_data = generate_opt_struct();
    c.bench_function("opt_struct_per_row", |b| {
        b.iter(|| {
            let df = opt_wrap_per_row(std::hint::black_box(&opt_data)).unwrap();
            std::hint::black_box(df)
        });
    });
    c.bench_function("opt_struct_bulk", |b| {
        b.iter(|| {
            let df =
                <OptWrap<Meta> as Columnar>::columnar_to_dataframe(std::hint::black_box(&opt_data))
                    .unwrap();
            std::hint::black_box(df)
        });
    });

    // Vec<T> direct: 1k parent rows × varying inner Vec lengths (1..=5 elements).
    let vec_data = generate_vec_struct();
    c.bench_function("vec_struct_per_row", |b| {
        b.iter(|| {
            let df = vec_wrap_per_row(std::hint::black_box(&vec_data)).unwrap();
            std::hint::black_box(df)
        });
    });
    c.bench_function("vec_struct_bulk", |b| {
        b.iter(|| {
            let df =
                <VecWrap<Meta> as Columnar>::columnar_to_dataframe(std::hint::black_box(&vec_data))
                    .unwrap();
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
