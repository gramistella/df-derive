// Regression: `#[derive(ToDataFrame)]` paired with `#[derive(Deserialize)]`
// must not trip `clippy::unsafe_derive_deserialize`.
//
// Earlier versions of the macro emitted unsafe list assembly inside the
// `ColumnarSpec::encode_columns` impl method on the user's struct. Clippy
// walks impl blocks of `Deserialize`-able types looking for `unsafe`, and
// that placement caused the lint to fire on downstream types. Generated list
// assembly now uses Polars' safe, dtype-checked constructor and contains no
// unsafe block.
//
// This file is compiled directly by `cargo build`/`cargo clippy` (not via
// `trybuild`), so the file-level `#![deny(...)]` actually fires under
// `cargo clippy` — `just lint` will fail if the macro ever inlines the
// `unsafe` back into an impl method on `Self`.
//
// Both bulk-emit shapes that need checked list assembly are exercised:
// - `Vec<DerivedStruct>` (the `gen_bulk_vec` path)
// - `Option<Vec<DerivedStruct>>` (the `gen_bulk_option_vec` path)
// — paired with a `Decimal` field, the shape that surfaced the original
// downstream report.
//
// Also covers the primitive-list paths that maintain their own value and
// validity buffers:
// - `Vec<i32>` exercises reserved numeric writes
// - `Vec<Option<i32>>` exercises prepared numeric validity writes
// - `Vec<Option<bool>>` exercises value and validity bitmaps
//
// `Option<String>` covers the direct-view validity path too. None of these
// generated paths may place an `unsafe` block inside the user's impl.

#![deny(clippy::unsafe_derive_deserialize)]

use df_derive::ToDataFrame;
use polars::prelude::*;
use rust_decimal::Decimal;
use serde::Deserialize;

use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Deserialize, Clone)]
struct Inner {
    field_a: i64,
    field_b: f64,
}

#[derive(ToDataFrame, Deserialize, Clone)]
struct Outer {
    id: u32,
    #[df_derive(decimal(precision = 18, scale = 6))]
    price: Decimal,
    #[df_derive(decimal(precision = 18, scale = 6))]
    maybe_price: Option<Decimal>,
    label: Option<String>,
    numbers: Vec<i32>,
    optional_numbers: Vec<Option<i32>>,
    flags: Vec<Option<bool>>,
    payloads: Vec<Inner>,
    optional_payloads: Option<Vec<Inner>>,
}

fn fixture_rows() -> Vec<Outer> {
    vec![
        Outer {
            id: 1,
            price: Decimal::new(12345, 2),
            maybe_price: Some(Decimal::new(6789, 2)),
            label: Some("alpha".to_string()),
            numbers: vec![1, -2, 3],
            optional_numbers: vec![Some(8), None, Some(-5)],
            flags: vec![Some(true), None, Some(false)],
            payloads: vec![
                Inner {
                    field_a: 10,
                    field_b: 1.5,
                },
                Inner {
                    field_a: 20,
                    field_b: 2.5,
                },
            ],
            optional_payloads: Some(vec![Inner {
                field_a: 30,
                field_b: 3.5,
            }]),
        },
        Outer {
            id: 2,
            price: Decimal::new(0, 0),
            maybe_price: None,
            label: None,
            numbers: vec![],
            optional_numbers: vec![None],
            flags: vec![None, Some(true)],
            payloads: vec![],
            optional_payloads: None,
        },
    ]
}

fn assert_i32_list(df: &DataFrame, column: &str, row: usize, expected: &[Option<i32>]) {
    let AnyValue::List(values) = df.column(column).unwrap().get(row).unwrap() else {
        panic!("expected {column} row {row} to be a list");
    };
    assert_eq!(values.i32().unwrap().iter().collect::<Vec<_>>(), expected);
}

fn assert_bool_list(df: &DataFrame, column: &str, row: usize, expected: &[Option<bool>]) {
    let AnyValue::List(values) = df.column(column).unwrap().get(row).unwrap() else {
        panic!("expected {column} row {row} to be a list");
    };
    assert_eq!(values.bool().unwrap().iter().collect::<Vec<_>>(), expected);
}

#[test]
fn derived_struct_with_deserialize_compiles_and_runs() {
    let rows = fixture_rows();

    let df_single = rows[0].to_dataframe().unwrap();
    assert_eq!(df_single.height(), 1);
    assert_eq!(
        df_single.column("price").unwrap().dtype(),
        &DataType::Decimal(18, 6)
    );
    assert_eq!(
        df_single.column("label").unwrap().dtype(),
        &DataType::String
    );
    assert_eq!(
        df_single.column("payloads.field_a").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Int64))
    );
    assert_eq!(
        df_single
            .column("optional_payloads.field_b")
            .unwrap()
            .dtype(),
        &DataType::List(Box::new(DataType::Float64))
    );

    let df_batch = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.height(), 2);
    assert_eq!(
        df_batch
            .column("optional_payloads.field_a")
            .unwrap()
            .dtype(),
        &DataType::List(Box::new(DataType::Int64))
    );
    assert_eq!(
        df_batch
            .column("optional_payloads.field_a")
            .unwrap()
            .get(1)
            .unwrap(),
        AnyValue::Null
    );
    assert_eq!(
        df_batch.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("alpha")
    );
    assert_eq!(
        df_batch.column("label").unwrap().get(1).unwrap(),
        AnyValue::Null
    );

    assert_i32_list(&df_batch, "numbers", 0, &[Some(1), Some(-2), Some(3)]);
    assert_i32_list(&df_batch, "numbers", 1, &[]);
    assert_i32_list(&df_batch, "optional_numbers", 0, &[Some(8), None, Some(-5)]);
    assert_bool_list(&df_batch, "flags", 0, &[Some(true), None, Some(false)]);
    assert_bool_list(&df_batch, "flags", 1, &[None, Some(true)]);
}
