// Regression: `#[derive(ToDataFrame)]` paired with `#[derive(Deserialize)]`
// must not trip `clippy::unsafe_derive_deserialize`.
//
// Earlier versions of the macro emitted `unsafe { Series::from_chunks_and_dtype_unchecked(..) }`
// directly inside the `Columnar::columnar_from_refs` impl method on the
// user's struct. Clippy walks impl blocks of `Deserialize`-able types
// looking for `unsafe`, and that placement caused the lint to fire on every
// `Decimal`-bearing struct downstream that paired
// `#[derive(ToDataFrame, Deserialize)]`. The fix hoists the `unsafe` call
// into a free helper function emitted at the top of the per-derive
// `const _: () = { ... };` scope; the lint now finds no `unsafe` inside the
// user type's impls.
//
// This file is compiled directly by `cargo build`/`cargo clippy` (not via
// `trybuild`), so the file-level `#![deny(...)]` actually fires under
// `cargo clippy` — `just lint` will fail if the macro ever inlines the
// `unsafe` back into an impl method on `Self`.
//
// Both bulk-emit shapes that use the unsafe call are exercised:
// - `Vec<DerivedStruct>` (the `gen_bulk_vec` path)
// - `Option<Vec<DerivedStruct>>` (the `gen_bulk_option_vec` path)
// — paired with a `Decimal` field, the shape that surfaced the original
// downstream report.
//
// Also covers `Option<String>` — the direct-view fast path for that shape
// uses a `MutableBitmap` for validity. An earlier draft used the unsafe
// `set_unchecked` to flip the per-row bit, which would have re-introduced
// `unsafe` into the user's impl method. The current code uses the safe
// `MutableBitmap::set` (bounds-checked, no `unsafe` keyword) so this lint
// does not fire on `Option<String>` fields either.

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
