// Regression: `Vec<Option<DerivedStruct>>` must surface inner-element
// `None` as strict `AnyValue::Null` per inner schema column, while the
// outer list itself stays non-null even for empty Vecs.
//
// The bulk emitter for this shape gathers `&Inner` references for each
// `Some(v)` element, calls `Inner::columnar_from_refs` once on the gathered
// slice, then expands each inner schema column via `Series::take(&IdxCa)`
// over a per-element position vector. A regression that flipped the
// per-element bit logic (treating `None` as `Some` or vice versa), or that
// swapped `validity = None` for the outer list, would either drop nulls or
// wreck offsets — both surface here as failed `AnyValue::Null` assertions.
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
    payload: Vec<Option<Inner>>,
}

fn list_dtype_for_field_a() -> DataType {
    DataType::List(Box::new(DataType::Int64))
}

fn list_dtype_for_field_b() -> DataType {
    DataType::List(Box::new(DataType::Float64))
}

fn assert_inner_columns_typed(df: &DataFrame, expected_height: usize) {
    assert_eq!(df.column("payload.field_a").unwrap().dtype(), &list_dtype_for_field_a());
    assert_eq!(df.column("payload.field_b").unwrap().dtype(), &list_dtype_for_field_b());
    assert_eq!(df.column("payload.field_a").unwrap().len(), expected_height);
    assert_eq!(df.column("payload.field_b").unwrap().len(), expected_height);
}

fn main() {
    test_empty_parent_slice();
    test_all_some_but_empty_vec();
    test_all_none_elements();
    test_mixed_some_none();
}

// Zero parents: the columnar path returns an empty DataFrame with the
// correct typed schema. The bulk emitter never enters its scan loop.
fn test_empty_parent_slice() {
    let rows: Vec<Outer> = Vec::new();
    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 0);
    assert_inner_columns_typed(&df, 0);
}

// Every parent has `Vec::new()`: outer offsets are all zero, the gathered
// slice is empty, every outer row must be a present-but-empty list (NOT
// null). Exercises the `total == 0` branch in the bulk emitter.
fn test_all_some_but_empty_vec() {
    let rows: Vec<Outer> = (0..4).map(|i| Outer { id: i, payload: Vec::new() }).collect();
    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 4);
    assert_inner_columns_typed(&df, 4);

    for idx in 0..4 {
        let av_a = df.column("payload.field_a").unwrap().get(idx).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(idx).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("row {idx} of payload.field_a must be a List(empty), got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("row {idx} of payload.field_b must be a List(empty), got {av_b:?}");
        };
        assert_eq!(s_a.len(), 0, "row {idx} payload.field_a must be empty list");
        assert_eq!(s_b.len(), 0, "row {idx} payload.field_b must be empty list");
    }
}

// Every parent has `vec![None, None, ...]`: outer offsets are non-zero
// but the gathered slice is empty. Exercises the all-absent branch in the
// bulk emitter — every inner-element slot must surface as a strict null
// inside its outer list.
fn test_all_none_elements() {
    let rows = vec![
        Outer { id: 0, payload: vec![None, None, None] },
        Outer { id: 1, payload: vec![None] },
        Outer { id: 2, payload: vec![None, None] },
    ];
    let expected_lens = [3usize, 1, 2];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 3);
    assert_inner_columns_typed(&df, 3);

    for (idx, &expected_len) in expected_lens.iter().enumerate() {
        let av_a = df.column("payload.field_a").unwrap().get(idx).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(idx).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("row {idx} payload.field_a must be a List, got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("row {idx} payload.field_b must be a List, got {av_b:?}");
        };
        assert_eq!(s_a.len(), expected_len, "row {idx} field_a outer list length");
        assert_eq!(s_b.len(), expected_len, "row {idx} field_b outer list length");
        for inner_idx in 0..expected_len {
            assert_eq!(
                s_a.get(inner_idx).unwrap(),
                AnyValue::Null,
                "row {idx} field_a inner element {inner_idx} must be Null"
            );
            assert_eq!(
                s_b.get(inner_idx).unwrap(),
                AnyValue::Null,
                "row {idx} field_b inner element {inner_idx} must be Null"
            );
        }
    }
}

// Mixed Some/None elements: each non-null inner value must round-trip
// correctly AND `None` positions inside the outer list must surface as
// strict `AnyValue::Null` for each inner field. Exercises the main
// `IdxCa` + `take` branch of the bulk emitter.
fn test_mixed_some_none() {
    let rows = vec![
        // [Some(a), None, Some(b)] → 3 outer slots, 2 valid + 1 null
        Outer {
            id: 0,
            payload: vec![
                Some(Inner { field_a: 10, field_b: 1.5 }),
                None,
                Some(Inner { field_a: 20, field_b: 2.5 }),
            ],
        },
        // Empty Vec — outer list present but empty.
        Outer { id: 1, payload: Vec::new() },
        // All None — outer list present, every inner null.
        Outer { id: 2, payload: vec![None, None] },
        // All Some.
        Outer {
            id: 3,
            payload: vec![
                Some(Inner { field_a: 30, field_b: 3.5 }),
                Some(Inner { field_a: 40, field_b: 4.5 }),
            ],
        },
        // Single None at a single slot.
        Outer { id: 4, payload: vec![None] },
        // Single Some.
        Outer { id: 5, payload: vec![Some(Inner { field_a: 50, field_b: 5.5 })] },
    ];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 6);
    assert_inner_columns_typed(&df, 6);

    // Per-row expectations: (expected outer-list length, expected
    // per-slot Option<value> per inner column).
    let expected_a: Vec<Vec<Option<i64>>> = vec![
        vec![Some(10), None, Some(20)],
        vec![],
        vec![None, None],
        vec![Some(30), Some(40)],
        vec![None],
        vec![Some(50)],
    ];
    let expected_b: Vec<Vec<Option<f64>>> = vec![
        vec![Some(1.5), None, Some(2.5)],
        vec![],
        vec![None, None],
        vec![Some(3.5), Some(4.5)],
        vec![None],
        vec![Some(5.5)],
    ];

    for (idx, (exp_a, exp_b)) in expected_a.iter().zip(expected_b.iter()).enumerate() {
        let av_a = df.column("payload.field_a").unwrap().get(idx).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(idx).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("row {idx} payload.field_a must be a List, got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("row {idx} payload.field_b must be a List, got {av_b:?}");
        };
        assert_eq!(s_a.len(), exp_a.len(), "row {idx} field_a outer list length");
        assert_eq!(s_b.len(), exp_b.len(), "row {idx} field_b outer list length");
        for (inner_idx, exp) in exp_a.iter().enumerate() {
            let av = s_a.get(inner_idx).unwrap();
            match exp {
                Some(v) => assert_eq!(
                    av,
                    AnyValue::Int64(*v),
                    "row {idx} field_a inner element {inner_idx} value mismatch"
                ),
                None => assert_eq!(
                    av,
                    AnyValue::Null,
                    "row {idx} field_a inner element {inner_idx} must be Null"
                ),
            }
        }
        for (inner_idx, exp) in exp_b.iter().enumerate() {
            let av = s_b.get(inner_idx).unwrap();
            match exp {
                Some(v) => assert_eq!(
                    av,
                    AnyValue::Float64(*v),
                    "row {idx} field_b inner element {inner_idx} value mismatch"
                ),
                None => assert_eq!(
                    av,
                    AnyValue::Null,
                    "row {idx} field_b inner element {inner_idx} must be Null"
                ),
            }
        }
    }
}
