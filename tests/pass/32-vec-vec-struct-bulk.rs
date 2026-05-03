// Regression: `Vec<Vec<DerivedStruct>>` must produce a list-of-lists per
// inner schema column, where the outer-list partitions per-outer-row and
// the inner-list partitions per-inner-Vec, with leaves drawn from the
// nested struct's columns.
//
// The bulk emitter for this shape flattens leaves into a single contiguous
// slice, calls `Inner::columnar_from_refs` exactly once, then stacks two
// `LargeListArray`s (inner-list + outer-list) per inner schema column. A
// regression in the offset bookkeeping would either drop leaves,
// mis-partition them between inner lists, or shift them into the wrong
// outer row — all surface here as failed `AnyValue` assertions or wrong
// list lengths.
//
// Schema parity: the declared schema and the runtime Series both carry
// `List<List<inner_dtype>>` — the schema generator wraps once per `Vec`
// wrapper depth (see also the assertion in `tests/pass/20-generics.rs`).
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
    payload: Vec<Vec<Inner>>,
}

// Both the populated path (bulk emitter wraps twice) and the empty-parent
// path (column dtype from `empty_dataframe()` which uses the schema-derived
// dtype) carry `List<List<leaf>>`.
fn nested_list_dtype_for_field_a() -> DataType {
    DataType::List(Box::new(DataType::List(Box::new(DataType::Int64))))
}

fn nested_list_dtype_for_field_b() -> DataType {
    DataType::List(Box::new(DataType::List(Box::new(DataType::Float64))))
}

fn assert_inner_columns_populated(df: &DataFrame, expected_height: usize) {
    assert_eq!(df.column("payload.field_a").unwrap().dtype(), &nested_list_dtype_for_field_a());
    assert_eq!(df.column("payload.field_b").unwrap().dtype(), &nested_list_dtype_for_field_b());
    assert_eq!(df.column("payload.field_a").unwrap().len(), expected_height);
    assert_eq!(df.column("payload.field_b").unwrap().len(), expected_height);
}

fn assert_inner_columns_empty_path(df: &DataFrame, expected_height: usize) {
    assert_eq!(df.column("payload.field_a").unwrap().dtype(), &nested_list_dtype_for_field_a());
    assert_eq!(df.column("payload.field_b").unwrap().dtype(), &nested_list_dtype_for_field_b());
    assert_eq!(df.column("payload.field_a").unwrap().len(), expected_height);
    assert_eq!(df.column("payload.field_b").unwrap().len(), expected_height);
}

fn main() {
    test_empty_parent_slice();
    test_all_outer_empty();
    test_outer_with_empty_inner_vecs();
    test_all_populated();
    test_mixed_shapes();
}

// Zero parents: the columnar path returns an empty DataFrame with the
// correct typed schema. The bulk emitter never enters its scan loop —
// `columnar_from_refs` short-circuits to `empty_dataframe()`, whose
// dtype comes from `Outer::schema()` (single `List<leaf>` wrap, since
// the schema generator only wraps once for `has_vec`).
fn test_empty_parent_slice() {
    let rows: Vec<Outer> = Vec::new();
    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 0);
    assert_inner_columns_empty_path(&df, 0);
}

// Every parent has `payload: Vec::new()`: outer offsets are all zero, no
// inner lists, no leaves. Every outer row must be a present-but-empty
// outer list. Exercises the fully-empty branch of the bulk emitter.
fn test_all_outer_empty() {
    let rows: Vec<Outer> = (0..4).map(|i| Outer { id: i, payload: Vec::new() }).collect();
    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 4);
    assert_inner_columns_populated(&df, 4);

    for idx in 0..4 {
        let av_a = df.column("payload.field_a").unwrap().get(idx).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(idx).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("row {idx} payload.field_a must be a List(empty), got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("row {idx} payload.field_b must be a List(empty), got {av_b:?}");
        };
        assert_eq!(s_a.len(), 0, "row {idx} payload.field_a must be empty list");
        assert_eq!(s_b.len(), 0, "row {idx} payload.field_b must be empty list");
    }
}

// Every parent has at least one inner Vec, but every inner Vec is empty.
// Exercises the `flat.is_empty()` branch with non-trivial outer offsets:
// the inner-list count is non-zero per row, but every inner list is empty.
fn test_outer_with_empty_inner_vecs() {
    let rows = vec![
        Outer { id: 0, payload: vec![Vec::new(), Vec::new()] },
        Outer { id: 1, payload: vec![Vec::new()] },
        Outer { id: 2, payload: vec![Vec::new(), Vec::new(), Vec::new()] },
    ];
    let expected_inner_lens = [2usize, 1, 3];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 3);
    assert_inner_columns_populated(&df, 3);

    for (idx, &expected_outer_len) in expected_inner_lens.iter().enumerate() {
        let av_a = df.column("payload.field_a").unwrap().get(idx).unwrap();
        let av_b = df.column("payload.field_b").unwrap().get(idx).unwrap();
        let AnyValue::List(s_a) = av_a else {
            panic!("row {idx} payload.field_a must be a List, got {av_a:?}");
        };
        let AnyValue::List(s_b) = av_b else {
            panic!("row {idx} payload.field_b must be a List, got {av_b:?}");
        };
        assert_eq!(s_a.len(), expected_outer_len, "row {idx} field_a outer list length");
        assert_eq!(s_b.len(), expected_outer_len, "row {idx} field_b outer list length");
        for inner_idx in 0..expected_outer_len {
            let av_inner_a = s_a.get(inner_idx).unwrap();
            let av_inner_b = s_b.get(inner_idx).unwrap();
            let AnyValue::List(inner_a) = av_inner_a else {
                panic!("row {idx} field_a inner element {inner_idx} must be a List(empty)");
            };
            let AnyValue::List(inner_b) = av_inner_b else {
                panic!("row {idx} field_b inner element {inner_idx} must be a List(empty)");
            };
            assert_eq!(inner_a.len(), 0, "row {idx} field_a inner {inner_idx} must be empty");
            assert_eq!(inner_b.len(), 0, "row {idx} field_b inner {inner_idx} must be empty");
        }
    }
}

// Every inner Vec has at least one element. Exercises the populated branch
// of the bulk emitter on a fully-dense input: per-leaf values must match,
// per-inner-list lengths must match, per-outer-row inner-list counts must
// match.
fn test_all_populated() {
    let rows = vec![
        Outer {
            id: 0,
            payload: vec![
                vec![Inner { field_a: 10, field_b: 1.5 }],
                vec![
                    Inner { field_a: 20, field_b: 2.5 },
                    Inner { field_a: 30, field_b: 3.5 },
                ],
            ],
        },
        Outer {
            id: 1,
            payload: vec![vec![
                Inner { field_a: 40, field_b: 4.5 },
                Inner { field_a: 50, field_b: 5.5 },
                Inner { field_a: 60, field_b: 6.5 },
            ]],
        },
    ];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 2);
    assert_inner_columns_populated(&df, 2);

    let expected_a: Vec<Vec<Vec<i64>>> =
        vec![vec![vec![10], vec![20, 30]], vec![vec![40, 50, 60]]];
    let expected_b: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![1.5], vec![2.5, 3.5]], vec![vec![4.5, 5.5, 6.5]]];

    for (idx, (exp_a, exp_b)) in expected_a.iter().zip(expected_b.iter()).enumerate() {
        verify_outer_row::<i64>(&df, idx, "payload.field_a", exp_a, |v| AnyValue::Int64(*v));
        verify_outer_row::<f64>(&df, idx, "payload.field_b", exp_b, |v| AnyValue::Float64(*v));
    }
}

// Mixed: empty outer, outer with all-empty inners, outer with mixed inner
// lengths, single-element inner. Pins all four outer-row patterns the
// emitter must handle: empty outer → no inner lists; empty inner → outer
// list contains an empty inner list; multi-element inner → leaves group
// correctly; partial mix → offsets stay aligned across all combinations.
fn test_mixed_shapes() {
    let rows = vec![
        // Empty outer (no inner lists).
        Outer { id: 0, payload: vec![] },
        // Outer with one populated inner.
        Outer {
            id: 1,
            payload: vec![vec![Inner { field_a: 100, field_b: 0.5 }]],
        },
        // Outer with one empty inner.
        Outer { id: 2, payload: vec![Vec::new()] },
        // Outer with mix of populated and empty inners.
        Outer {
            id: 3,
            payload: vec![
                vec![Inner { field_a: 200, field_b: 1.5 }],
                Vec::new(),
                vec![
                    Inner { field_a: 300, field_b: 2.5 },
                    Inner { field_a: 400, field_b: 3.5 },
                ],
            ],
        },
        // Empty outer again, after populated rows.
        Outer { id: 4, payload: vec![] },
        // Single inner with multiple leaves.
        Outer {
            id: 5,
            payload: vec![vec![
                Inner { field_a: 500, field_b: 4.5 },
                Inner { field_a: 600, field_b: 5.5 },
            ]],
        },
    ];

    let df = <Outer as Columnar>::columnar_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 6);
    assert_inner_columns_populated(&df, 6);

    let expected_a: Vec<Vec<Vec<i64>>> = vec![
        vec![],
        vec![vec![100]],
        vec![vec![]],
        vec![vec![200], vec![], vec![300, 400]],
        vec![],
        vec![vec![500, 600]],
    ];
    let expected_b: Vec<Vec<Vec<f64>>> = vec![
        vec![],
        vec![vec![0.5]],
        vec![vec![]],
        vec![vec![1.5], vec![], vec![2.5, 3.5]],
        vec![],
        vec![vec![4.5, 5.5]],
    ];

    for (idx, (exp_a, exp_b)) in expected_a.iter().zip(expected_b.iter()).enumerate() {
        verify_outer_row::<i64>(&df, idx, "payload.field_a", exp_a, |v| AnyValue::Int64(*v));
        verify_outer_row::<f64>(&df, idx, "payload.field_b", exp_b, |v| AnyValue::Float64(*v));
    }
}

fn verify_outer_row<T>(
    df: &DataFrame,
    idx: usize,
    col: &str,
    expected: &[Vec<T>],
    leaf_to_av: impl Fn(&T) -> AnyValue<'static>,
) {
    let av = df.column(col).unwrap().get(idx).unwrap();
    let AnyValue::List(outer_s) = av else {
        panic!("row {idx} {col} must be a List, got {av:?}");
    };
    assert_eq!(outer_s.len(), expected.len(), "row {idx} {col} outer list length");
    for (inner_idx, exp_inner) in expected.iter().enumerate() {
        let av_inner = outer_s.get(inner_idx).unwrap();
        let AnyValue::List(inner_s) = av_inner else {
            panic!("row {idx} {col} inner element {inner_idx} must be a List, got {av_inner:?}");
        };
        assert_eq!(
            inner_s.len(),
            exp_inner.len(),
            "row {idx} {col} inner {inner_idx} length"
        );
        for (leaf_idx, exp_leaf) in exp_inner.iter().enumerate() {
            let av_leaf = inner_s.get(leaf_idx).unwrap();
            assert_eq!(
                av_leaf,
                leaf_to_av(exp_leaf),
                "row {idx} {col} inner {inner_idx} leaf {leaf_idx} value mismatch"
            );
        }
    }
}
