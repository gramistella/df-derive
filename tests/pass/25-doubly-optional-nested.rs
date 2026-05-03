// Concrete-struct `Option<Option<Inner>>` coverage.
//
// `[]`, `[Option]`, and `[Vec]` shapes around a nested struct are served by
// the bulk encoder fast paths. `Option<Option<Inner>>` falls through to the
// per-row push pipeline, which is the only path that exercises the trait
// `to_dataframe()` indirection across struct boundaries for a concrete inner
// struct, so the shape is worth a dedicated regression test. The generic
// equivalent already lives in `20-generics.rs` (`OptOptWrapper<T>`); this
// file is the non-generic mirror.
//
// Polars cannot represent two distinct null states (the outer `None` and the
// inner `None`), so `Some(None)` and `None` collapse to the same `AnyValue::Null`
// at the schema level. That's the documented contract.

use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{Columnar, ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct Inner {
    value: f64,
    label: String,
}

#[derive(ToDataFrame, Clone)]
struct OptOptInner {
    id: u32,
    payload: Option<Option<Inner>>,
}

fn main() {
    let expected_schema = vec![
        ("id".into(), DataType::UInt32),
        ("payload.value".into(), DataType::Float64),
        ("payload.label".into(), DataType::String),
    ];
    assert_eq!(OptOptInner::schema().unwrap(), expected_schema);

    let items = vec![
        OptOptInner {
            id: 1,
            payload: Some(Some(Inner {
                value: 7.5,
                label: "alpha".into(),
            })),
        },
        OptOptInner {
            id: 2,
            payload: Some(None),
        },
        OptOptInner {
            id: 3,
            payload: None,
        },
    ];

    let batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape(), (3, 3));
    assert_eq!(batch.get_column_names(), vec!["id", "payload.value", "payload.label"]);

    assert_eq!(
        batch.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt32(1)
    );
    assert_eq!(
        batch.column("payload.value").unwrap().get(0).unwrap(),
        AnyValue::Float64(7.5)
    );
    let label0 = batch.column("payload.label").unwrap().get(0).unwrap();
    if let AnyValue::String(s) = label0 {
        assert_eq!(s, "alpha");
    } else {
        panic!("expected string AnyValue, got {label0:?}");
    }

    // Some(None) and None both collapse to nulls in every payload column.
    for row in [1, 2] {
        assert!(matches!(
            batch.column("payload.value").unwrap().get(row).unwrap(),
            AnyValue::Null
        ));
        assert!(matches!(
            batch.column("payload.label").unwrap().get(row).unwrap(),
            AnyValue::Null
        ));
    }

    // Single-row API exercises the same on-leaf path through
    // `to_dataframe(&self) -> Columnar::columnar_from_refs(&[self])`.
    let single = OptOptInner {
        id: 42,
        payload: Some(Some(Inner {
            value: 99.25,
            label: "solo".into(),
        })),
    };
    let single_df = single.to_dataframe().unwrap();
    assert_eq!(single_df.shape(), (1, 3));
    assert_eq!(
        single_df.column("payload.value").unwrap().get(0).unwrap(),
        AnyValue::Float64(99.25)
    );

    // Empty slice and empty_dataframe both produce a zero-row frame with
    // the declared schema.
    let empty_via_slice = <[OptOptInner] as ToDataFrameVec>::to_dataframe(&[]).unwrap();
    assert_eq!(empty_via_slice.shape(), (0, 3));

    let empty_via_columnar = <OptOptInner as Columnar>::columnar_to_dataframe(&[]).unwrap();
    assert_eq!(empty_via_columnar.shape(), (0, 3));

    let empty_direct = OptOptInner::empty_dataframe().unwrap();
    assert_eq!(empty_direct.shape(), (0, 3));
}
