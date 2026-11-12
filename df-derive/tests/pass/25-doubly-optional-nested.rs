// `Option<Option<Inner>>` around a concrete nested struct.
//
// A column has one validity bit per row, so the outer `None` and the inner
// `None` cannot both be kept: `Some(None)` and `None` are the same null in
// every column of the nested struct. That is the documented contract. The
// generic counterpart is `OptOptWrapper<T>` in `20-generics.rs`.

use df_derive::ToDataFrame;
use polars::prelude::*;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};

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

    // `Some(None)` and `None` are both null in every payload column.
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

    // One value converts the same way as a slice of them.
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

    let empty_via_columnar = <OptOptInner as ToDataFrame>::rows_to_dataframe(&[]).unwrap();
    assert_eq!(empty_via_columnar.shape(), (0, 3));

    let empty_direct = OptOptInner::empty_dataframe().unwrap();
    assert_eq!(empty_direct.shape(), (0, 3));
}
