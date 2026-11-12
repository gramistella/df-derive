// Regression: `#[derive(ToDataFrame)]` paired with `#[derive(Deserialize)]`
// must not trip `clippy::unsafe_derive_deserialize`.
//
// Clippy walks the impl blocks of `Deserialize`-able types looking for
// `unsafe`. Early versions of the macro emitted an `unsafe` block inside an
// impl on the user's struct, which fired the lint on every downstream type
// deriving both traits. Generated code now contains no `unsafe` at all, and
// neither does the runtime it calls.
//
// This file is compiled directly by `cargo build`/`cargo clippy` (not via
// `trybuild`), so the file-level `#![deny(...)]` fires under `just lint` if
// the derive ever emits `unsafe` into an impl on `Self` again. The shapes
// below are the ones that used to need it: lists of derived structs, an
// optional list of them, a decimal field and an optional string.

#![deny(clippy::unsafe_derive_deserialize)]

use df_derive::ToDataFrame;
use polars::prelude::*;
use rust_decimal::Decimal;
use serde::Deserialize;

#[path = "common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

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
    payloads: Vec<Inner>,
    optional_payloads: Option<Vec<Inner>>,
}

#[test]
fn derived_struct_with_deserialize_compiles_and_runs() {
    let rows = vec![
        Outer {
            id: 1,
            price: Decimal::new(12345, 2),
            maybe_price: Some(Decimal::new(6789, 2)),
            label: Some("alpha".to_string()),
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
            payloads: vec![],
            optional_payloads: None,
        },
    ];

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
}
