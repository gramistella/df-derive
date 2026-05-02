// Regression: `Option<Vec<Option<DerivedStruct>>>` must distinguish three
// outer states (None / Some(empty) / Some(non-empty)) AND surface inner
// `None` elements as strict `AnyValue::Null` per inner schema column.
//
// The bulk emitter for this shape (`gen_bulk_option_vec_option`) fuses two
// patterns:
//   - the validity-bitmap outer-list pattern from `gen_bulk_option_vec`,
//     which makes `None` outer rows distinct from `Some(vec![])`,
//   - the per-element `IdxCa` scatter from `gen_bulk_vec_option`, which
//     materializes inner `None` slots as null inside present outer lists.
//
// A regression in either half (dropped/inverted bitmap bits, mis-ordered
// `pos` pushes, swapped offset deltas) collapses cases that must stay
// distinct — this file pins all four runtime branches:
// `total == 0`, `flat.is_empty() && total > 0`, `flat.len() == total`,
// and the mixed `IdxCa` + `take` branch.
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
    payload: Option<Vec<Option<Inner>>>,
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
    test_all_none_outer();
    test_all_some_empty_outer();
    test_all_some_inner_all_none();
    test_mixed_all_branches();
}

// Zero parents: the columnar path returns an empty DataFrame with the
// correct typed schema. The bulk emitter never enters its scan loop.
fn test_empty_parent_slice() {
    let rows: Vec<Outer> = Vec::new();
    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 0);
    assert_inner_columns_typed(&df, 0);
}

// Every parent has `payload: None`: outer offsets are all zero, the gathered
// slice is empty, validity bitmap is all-false. Every outer row must surface
// as `AnyValue::Null` (NOT empty list). Exercises the `total == 0` branch
// with all-null bitmap.
fn test_all_none_outer() {
    let rows: Vec<Outer> = (0..4).map(|i| Outer { id: i, payload: None }).collect();
    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 4);
    assert_inner_columns_typed(&df, 4);

    for idx in 0..4 {
        assert_eq!(
            df.column("payload.field_a").unwrap().get(idx).unwrap(),
            AnyValue::Null,
            "row {idx} of payload.field_a must be Null"
        );
        assert_eq!(
            df.column("payload.field_b").unwrap().get(idx).unwrap(),
            AnyValue::Null,
            "row {idx} of payload.field_b must be Null"
        );
    }
}

// Every parent has `payload: Some(vec![])`: outer validity is all-true,
// offsets all zero, gathered slice empty. Every outer row must surface as
// `AnyValue::List(empty)` (NOT Null). Same `total == 0` branch as above but
// with all-true bitmap.
fn test_all_some_empty_outer() {
    let rows: Vec<Outer> = (0..4)
        .map(|i| Outer {
            id: i,
            payload: Some(Vec::new()),
        })
        .collect();
    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 4);
    assert_inner_columns_typed(&df, 4);

    for idx in 0..4 {
        let av_a = df.column("payload.field_a").unwrap().get(idx).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(idx).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("Some(vec![]) row {idx} of payload.field_a must be a List(empty), got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("Some(vec![]) row {idx} of payload.field_b must be a List(empty), got {av_b:?}");
        };
        assert_eq!(s_a.len(), 0, "row {idx} payload.field_a must be empty list");
        assert_eq!(s_b.len(), 0, "row {idx} payload.field_b must be empty list");
    }
}

// Every parent has `payload: Some(vec![None, None, ...])`: outer validity
// is all-true, offsets non-zero, gathered slice empty. Inner positions are
// all None. Exercises the `flat.is_empty() && total > 0` branch — every
// inner-element slot must surface as a strict null inside its non-null
// outer list.
fn test_all_some_inner_all_none() {
    let rows = vec![
        Outer { id: 0, payload: Some(vec![None, None, None]) },
        Outer { id: 1, payload: Some(vec![None]) },
        Outer { id: 2, payload: Some(vec![None, None]) },
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

// Mixed: covers all three outer states (None / Some(empty) / Some(...))
// and both inner states (Some / None). Distinct outer-row + inner-element
// combinations are pinned to ensure the bitmap bits, offsets, and pos
// vector stay in sync. Exercises the main `flat.len() == total` and
// `IdxCa` + `take` branches.
fn test_mixed_all_branches() {
    let rows = vec![
        // Some(non-empty, all Some) — exercises the `flat.len() == total` branch
        // when this row is the only Some-with-content row.
        Outer {
            id: 0,
            payload: Some(vec![
                Some(Inner { field_a: 10, field_b: 1.5 }),
                Some(Inner { field_a: 20, field_b: 2.5 }),
            ]),
        },
        // None outer — bitmap bit false, offset delta 0.
        Outer { id: 1, payload: None },
        // Some(vec![]) — bitmap bit true, offset delta 0.
        Outer { id: 2, payload: Some(Vec::new()) },
        // Some([None]) — bitmap bit true, offset delta 1, all inner None.
        Outer { id: 3, payload: Some(vec![None]) },
        // Some([Some, None, Some]) — exercises mixed `IdxCa` + `take`.
        Outer {
            id: 4,
            payload: Some(vec![
                Some(Inner { field_a: 30, field_b: 3.5 }),
                None,
                Some(Inner { field_a: 40, field_b: 4.5 }),
            ]),
        },
        // None again — make sure repeated None doesn't drift bitmap.
        Outer { id: 5, payload: None },
        // Some([Some]) — single-element non-null list.
        Outer {
            id: 6,
            payload: Some(vec![Some(Inner { field_a: 50, field_b: 5.5 })]),
        },
    ];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 7);
    assert_inner_columns_typed(&df, 7);

    let expected_a: Vec<Option<Vec<Option<i64>>>> = vec![
        Some(vec![Some(10), Some(20)]),
        None,
        Some(vec![]),
        Some(vec![None]),
        Some(vec![Some(30), None, Some(40)]),
        None,
        Some(vec![Some(50)]),
    ];
    let expected_b: Vec<Option<Vec<Option<f64>>>> = vec![
        Some(vec![Some(1.5), Some(2.5)]),
        None,
        Some(vec![]),
        Some(vec![None]),
        Some(vec![Some(3.5), None, Some(4.5)]),
        None,
        Some(vec![Some(5.5)]),
    ];

    for (idx, (exp_a, exp_b)) in expected_a.iter().zip(expected_b.iter()).enumerate() {
        let av_a = df.column("payload.field_a").unwrap().get(idx).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(idx).unwrap();
        match exp_a {
            None => assert_eq!(
                av_a,
                AnyValue::Null,
                "row {idx} payload.field_a must be Null (outer None)"
            ),
            Some(slots) => {
                let AnyValue::List(s_a) = av_a else {
                    panic!("row {idx} payload.field_a must be a List, got {av_a:?}");
                };
                assert_eq!(s_a.len(), slots.len(), "row {idx} field_a outer list length");
                for (inner_idx, slot) in slots.iter().enumerate() {
                    let av = s_a.get(inner_idx).unwrap();
                    match slot {
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
            }
        }
        match exp_b {
            None => assert_eq!(
                av_b,
                AnyValue::Null,
                "row {idx} payload.field_b must be Null (outer None)"
            ),
            Some(slots) => {
                let AnyValue::List(s_b) = av_b else {
                    panic!("row {idx} payload.field_b must be a List, got {av_b:?}");
                };
                assert_eq!(s_b.len(), slots.len(), "row {idx} field_b outer list length");
                for (inner_idx, slot) in slots.iter().enumerate() {
                    let av = s_b.get(inner_idx).unwrap();
                    match slot {
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
    }
}
