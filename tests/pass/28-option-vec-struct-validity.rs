// Regression: `Option<Vec<DerivedStruct>>` None-rows must surface as
// `AnyValue::Null`, not as an empty list.
//
// The bulk emitter for this shape (`gen_bulk_option_vec`) builds a
// `LargeListArray` with a validity bitmap so that a `None` outer row is
// distinct from `Some(vec![])`. A regression that pushed `true` into the
// bitmap for `None` rows — or dropped the bitmap entirely — would collapse
// both cases to an empty list. The two existing checks at higher levels are
// too lenient (`AnyValue::Null | AnyValue::List(_)` is allowed for None
// rows), so this file pins the strict semantics: `None` ⇒ `AnyValue::Null`,
// `Some(vec![])` ⇒ `AnyValue::List(empty)`.
//
// We use `Columnar::columnar_to_dataframe` directly because the bulk
// emitters are invoked from that path; the per-row pipeline takes a
// different code path that this test is not trying to cover.

use df_derive::ToDataFrame;
use polars::prelude::*;
use pretty_assertions::assert_eq;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::Columnar;

#[derive(ToDataFrame, Clone)]
struct Inner {
    field_a: i64,
    field_b: f64,
}

#[derive(ToDataFrame, Clone)]
struct Outer {
    id: u32,
    payload: Option<Vec<Inner>>,
}

const EXPECTED_HEIGHT: usize = 5;

fn list_dtype_for_field_a() -> DataType {
    DataType::List(Box::new(DataType::Int64))
}

fn list_dtype_for_field_b() -> DataType {
    DataType::List(Box::new(DataType::Float64))
}

fn assert_inner_columns_are_null_lists(df: &DataFrame) {
    assert_eq!(df.column("payload.field_a").unwrap().dtype(), &list_dtype_for_field_a());
    assert_eq!(df.column("payload.field_b").unwrap().dtype(), &list_dtype_for_field_b());
    assert_eq!(df.column("payload.field_a").unwrap().len(), EXPECTED_HEIGHT);
    assert_eq!(df.column("payload.field_b").unwrap().len(), EXPECTED_HEIGHT);
}

fn main() {
    test_all_none();
    test_mixed_some_none();
    test_some_empty_vs_none();
}

fn test_all_none() {
    let rows: Vec<Outer> = (0..EXPECTED_HEIGHT as u32)
        .map(|i| Outer {
            id: i,
            payload: None,
        })
        .collect();

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), EXPECTED_HEIGHT);
    assert_inner_columns_are_null_lists(&df);

    for idx in 0..EXPECTED_HEIGHT {
        assert_eq!(
            df.column("payload.field_a").unwrap().get(idx).unwrap(),
            AnyValue::Null,
            "row {idx} of payload.field_a must be Null, not an empty list"
        );
        assert_eq!(
            df.column("payload.field_b").unwrap().get(idx).unwrap(),
            AnyValue::Null,
            "row {idx} of payload.field_b must be Null, not an empty list"
        );
    }
}

fn test_mixed_some_none() {
    let some_payload = || {
        Some(vec![
            Inner {
                field_a: 10,
                field_b: 1.5,
            },
            Inner {
                field_a: 20,
                field_b: 2.5,
            },
        ])
    };

    let rows = vec![
        Outer {
            id: 0,
            payload: some_payload(),
        },
        Outer {
            id: 1,
            payload: None,
        },
        Outer {
            id: 2,
            payload: some_payload(),
        },
        Outer {
            id: 3,
            payload: None,
        },
        Outer {
            id: 4,
            payload: some_payload(),
        },
    ];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), EXPECTED_HEIGHT);
    assert_inner_columns_are_null_lists(&df);

    for null_row in [1, 3] {
        assert_eq!(
            df.column("payload.field_a").unwrap().get(null_row).unwrap(),
            AnyValue::Null,
            "None row {null_row} of payload.field_a must be Null"
        );
        assert_eq!(
            df.column("payload.field_b").unwrap().get(null_row).unwrap(),
            AnyValue::Null,
            "None row {null_row} of payload.field_b must be Null"
        );
    }

    for some_row in [0, 2, 4] {
        let av_a = df.column("payload.field_a").unwrap().get(some_row).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(some_row).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("Some row {some_row} of payload.field_a must be a List, got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("Some row {some_row} of payload.field_b must be a List, got {av_b:?}");
        };
        assert_eq!(s_a.len(), 2);
        assert_eq!(s_b.len(), 2);
        let a_vec: Vec<i64> = s_a.i64().unwrap().into_no_null_iter().collect();
        let b_vec: Vec<f64> = s_b.f64().unwrap().into_no_null_iter().collect();
        assert_eq!(a_vec, vec![10, 20]);
        assert_eq!(b_vec, vec![1.5, 2.5]);
    }
}

fn test_some_empty_vs_none() {
    // Rows 0, 2 → Some(vec![]) (empty list, not null)
    // Rows 1, 3 → None         (strict null)
    // Row  4    → Some(vec![Inner{..}]) so the flat path actually fires
    let rows = vec![
        Outer {
            id: 0,
            payload: Some(vec![]),
        },
        Outer {
            id: 1,
            payload: None,
        },
        Outer {
            id: 2,
            payload: Some(vec![]),
        },
        Outer {
            id: 3,
            payload: None,
        },
        Outer {
            id: 4,
            payload: Some(vec![Inner {
                field_a: 99,
                field_b: 9.9,
            }]),
        },
    ];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), EXPECTED_HEIGHT);
    assert_inner_columns_are_null_lists(&df);

    for none_row in [1, 3] {
        assert_eq!(
            df.column("payload.field_a").unwrap().get(none_row).unwrap(),
            AnyValue::Null,
            "None row {none_row} must collapse to Null, not empty list"
        );
        assert_eq!(
            df.column("payload.field_b").unwrap().get(none_row).unwrap(),
            AnyValue::Null,
            "None row {none_row} must collapse to Null, not empty list"
        );
    }

    for empty_row in [0, 2] {
        let av_a = df.column("payload.field_a").unwrap().get(empty_row).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(empty_row).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("Some(vec![]) row {empty_row} of payload.field_a must be List(empty), got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("Some(vec![]) row {empty_row} of payload.field_b must be List(empty), got {av_b:?}");
        };
        assert_eq!(s_a.len(), 0, "Some(vec![]) row {empty_row} must be an empty list");
        assert_eq!(s_b.len(), 0, "Some(vec![]) row {empty_row} must be an empty list");
    }

    let av_a = df.column("payload.field_a").unwrap().get(4).unwrap();
    let av_b = df.column("payload.field_b").unwrap().get(4).unwrap();
    let AnyValue::List(s_a) = av_a else {
        panic!("Some(non-empty) row 4 of payload.field_a must be a List, got {av_a:?}");
    };
    let AnyValue::List(s_b) = av_b else {
        panic!("Some(non-empty) row 4 of payload.field_b must be a List, got {av_b:?}");
    };
    let a_vec: Vec<i64> = s_a.i64().unwrap().into_no_null_iter().collect();
    let b_vec: Vec<f64> = s_b.f64().unwrap().into_no_null_iter().collect();
    assert_eq!(a_vec, vec![99]);
    assert_eq!(b_vec, vec![9.9]);
}
