// `Option<Vec<Option<Struct>>>`: the list has three states (`None`,
// `Some(vec![])`, `Some` with elements) and each element has two. All of
// them must stay distinct in every column of the nested struct.
//
// Two validities are at work. The outer option decides whether the list is
// null, which is what separates `None` from `Some(vec![])`. The inner option
// decides whether an element is null inside a list that is present. The
// cases below hold one of them fixed and vary the other, then mix both.

use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrame;
use polars::prelude::*;
use pretty_assertions::assert_eq;

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
    assert_eq!(
        df.column("payload.field_a").unwrap().dtype(),
        &list_dtype_for_field_a()
    );
    assert_eq!(
        df.column("payload.field_b").unwrap().dtype(),
        &list_dtype_for_field_b()
    );
    assert_eq!(df.column("payload.field_a").unwrap().len(), expected_height);
    assert_eq!(df.column("payload.field_b").unwrap().len(), expected_height);
}

#[test]
fn runtime_semantics() {
    test_empty_parent_slice();
    test_all_none_outer();
    test_all_some_empty_outer();
    test_all_some_inner_all_none();
    test_mixed_all_branches();
}

// No rows: an empty frame with the full schema.
fn test_empty_parent_slice() {
    let rows: Vec<Outer> = Vec::new();
    let df = <Outer as ToDataFrame>::rows_to_dataframe(&rows).unwrap();
    assert_eq!(df.height(), 0);
    assert_inner_columns_typed(&df, 0);
}

// Every row has `payload: None`: every list is null, not empty.
fn test_all_none_outer() {
    let rows: Vec<Outer> = (0..4)
        .map(|i| Outer {
            id: i,
            payload: None,
        })
        .collect();
    let df = <Outer as ToDataFrame>::rows_to_dataframe(&rows).unwrap();
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

// Every row has `payload: Some(vec![])`: every list is present and empty,
// not null.
fn test_all_some_empty_outer() {
    let rows: Vec<Outer> = (0..4)
        .map(|i| Outer {
            id: i,
            payload: Some(Vec::new()),
        })
        .collect();
    let df = <Outer as ToDataFrame>::rows_to_dataframe(&rows).unwrap();
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

// Every row has `payload: Some(vec![None, None, ...])`: the lists are
// present and have length, and every element in them is null.
fn test_all_some_inner_all_none() {
    let rows = vec![
        Outer {
            id: 0,
            payload: Some(vec![None, None, None]),
        },
        Outer {
            id: 1,
            payload: Some(vec![None]),
        },
        Outer {
            id: 2,
            payload: Some(vec![None, None]),
        },
    ];
    let expected_lens = [3usize, 1, 2];

    let df = <Outer as ToDataFrame>::rows_to_dataframe(&rows).unwrap();
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
        assert_eq!(
            s_a.len(),
            expected_len,
            "row {idx} field_a outer list length"
        );
        assert_eq!(
            s_b.len(),
            expected_len,
            "row {idx} field_b outer list length"
        );
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

// All three list states and both element states in one frame, in an order
// that would expose a validity flag or a list boundary that is off by one.
fn test_mixed_all_branches() {
    let rows = vec![
        // Some(non-empty, all Some).
        Outer {
            id: 0,
            payload: Some(vec![
                Some(Inner {
                    field_a: 10,
                    field_b: 1.5,
                }),
                Some(Inner {
                    field_a: 20,
                    field_b: 2.5,
                }),
            ]),
        },
        // None: a null list, no elements.
        Outer {
            id: 1,
            payload: None,
        },
        // Some(vec![]): a present list, no elements.
        Outer {
            id: 2,
            payload: Some(Vec::new()),
        },
        // Some([None]): a present list of one null element.
        Outer {
            id: 3,
            payload: Some(vec![None]),
        },
        // Some([Some, None, Some]): values and a null in one list.
        Outer {
            id: 4,
            payload: Some(vec![
                Some(Inner {
                    field_a: 30,
                    field_b: 3.5,
                }),
                None,
                Some(Inner {
                    field_a: 40,
                    field_b: 4.5,
                }),
            ]),
        },
        // None again: a second null list must not shift the flags after it.
        Outer {
            id: 5,
            payload: None,
        },
        // Some([Some]): one element, not null.
        Outer {
            id: 6,
            payload: Some(vec![Some(Inner {
                field_a: 50,
                field_b: 5.5,
            })]),
        },
    ];

    let df = <Outer as ToDataFrame>::rows_to_dataframe(&rows).unwrap();
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
                assert_eq!(
                    s_a.len(),
                    slots.len(),
                    "row {idx} field_a outer list length"
                );
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
                assert_eq!(
                    s_b.len(),
                    slots.len(),
                    "row {idx} field_b outer list length"
                );
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
