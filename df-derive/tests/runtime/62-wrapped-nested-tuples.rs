use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrameVec;
use polars::prelude::*;

#[derive(ToDataFrame)]
struct WrappedParent {
    nested: Option<((i32, String), bool)>,
}

#[derive(ToDataFrame)]
struct WrappedElement {
    nested: (Option<(i32, String)>, bool),
}

#[derive(ToDataFrame)]
struct WrappedVec {
    nested: Vec<((i32, String), bool)>,
}

#[derive(ToDataFrame)]
#[allow(clippy::type_complexity)]
struct ComposedWrappers {
    nested: Option<Vec<(Option<(i32, String)>, bool, Vec<(i32, String)>)>>,
}

#[derive(ToDataFrame)]
struct NestedPayload {
    id: i32,
    label: String,
}

#[derive(ToDataFrame)]
struct AllAbsentNestedPayloads {
    nested: Vec<(Option<NestedPayload>, bool)>,
}

fn string_values(column: &Column) -> Vec<Option<String>> {
    column
        .str()
        .unwrap()
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect()
}

fn list_i32s(value: AnyValue<'_>) -> Vec<Option<i32>> {
    match value {
        AnyValue::List(series) => series.i32().unwrap().iter().collect(),
        other => panic!("expected i32 List, got {other:?}"),
    }
}

fn list_strings(value: AnyValue<'_>) -> Vec<Option<String>> {
    match value {
        AnyValue::List(series) => series
            .str()
            .unwrap()
            .iter()
            .map(|value| value.map(str::to_owned))
            .collect(),
        other => panic!("expected string List, got {other:?}"),
    }
}

fn list_bools(value: AnyValue<'_>) -> Vec<Option<bool>> {
    match value {
        AnyValue::List(series) => series.bool().unwrap().iter().collect(),
        other => panic!("expected bool List, got {other:?}"),
    }
}

fn nested_i32_lists(value: AnyValue<'_>) -> Vec<Option<Vec<Option<i32>>>> {
    let AnyValue::List(outer) = value else {
        panic!("expected outer List, got {value:?}");
    };
    (0..outer.len())
        .map(|idx| match outer.get(idx).unwrap() {
            AnyValue::List(inner) => Some(inner.i32().unwrap().iter().collect()),
            AnyValue::Null => None,
            other => panic!("expected inner i32 List or Null, got {other:?}"),
        })
        .collect()
}

fn nested_string_lists(value: AnyValue<'_>) -> Vec<Option<Vec<Option<String>>>> {
    let AnyValue::List(outer) = value else {
        panic!("expected outer List, got {value:?}");
    };
    (0..outer.len())
        .map(|idx| match outer.get(idx).unwrap() {
            AnyValue::List(inner) => Some(
                inner
                    .str()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(str::to_owned))
                    .collect(),
            ),
            AnyValue::Null => None,
            other => panic!("expected inner string List or Null, got {other:?}"),
        })
        .collect()
}

fn assert_nested_column_names(df: &DataFrame) {
    assert_eq!(
        df.get_column_names(),
        [
            "nested.field_0.field_0",
            "nested.field_0.field_1",
            "nested.field_1",
        ]
    );
}

#[test]
fn option_wrapped_nested_tuple_parent_preserves_null_rows() {
    let rows = [
        WrappedParent {
            nested: Some(((1, "p1".to_owned()), true)),
        },
        WrappedParent { nested: None },
        WrappedParent {
            nested: Some(((2, "p2".to_owned()), false)),
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(df.shape(), (3, 3));
    assert_nested_column_names(&df);
    assert_eq!(
        df.column("nested.field_0.field_0")
            .unwrap()
            .i32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1), None, Some(2)]
    );
    assert_eq!(
        string_values(df.column("nested.field_0.field_1").unwrap()),
        vec![Some("p1".to_owned()), None, Some("p2".to_owned())]
    );
    assert_eq!(
        df.column("nested.field_1")
            .unwrap()
            .bool()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(true), None, Some(false)]
    );
}

#[test]
fn option_wrapped_nested_tuple_element_does_not_null_its_sibling() {
    let rows = [
        WrappedElement {
            nested: (Some((3, "e1".to_owned())), true),
        },
        WrappedElement {
            nested: (None, false),
        },
        WrappedElement {
            nested: (Some((4, "e2".to_owned())), false),
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(df.shape(), (3, 3));
    assert_nested_column_names(&df);
    assert_eq!(
        df.column("nested.field_0.field_0")
            .unwrap()
            .i32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(3), None, Some(4)]
    );
    assert_eq!(
        string_values(df.column("nested.field_0.field_1").unwrap()),
        vec![Some("e1".to_owned()), None, Some("e2".to_owned())]
    );
    assert_eq!(
        df.column("nested.field_1")
            .unwrap()
            .bool()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(true), Some(false), Some(false)]
    );
}

#[test]
fn vec_wrapped_nested_tuple_preserves_nonempty_empty_nonempty_offsets() {
    let rows = [
        WrappedVec {
            nested: vec![((5, "v1".to_owned()), true), ((6, "v2".to_owned()), false)],
        },
        WrappedVec { nested: Vec::new() },
        WrappedVec {
            nested: vec![((7, "v3".to_owned()), true)],
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(df.shape(), (3, 3));
    assert_nested_column_names(&df);
    assert_eq!(
        list_i32s(df.column("nested.field_0.field_0").unwrap().get(0).unwrap()),
        vec![Some(5), Some(6)]
    );
    assert_eq!(
        list_strings(df.column("nested.field_0.field_1").unwrap().get(0).unwrap()),
        vec![Some("v1".to_owned()), Some("v2".to_owned())]
    );
    assert_eq!(
        list_bools(df.column("nested.field_1").unwrap().get(0).unwrap()),
        vec![Some(true), Some(false)]
    );

    for column in df.columns() {
        let AnyValue::List(empty) = column.get(1).unwrap() else {
            panic!(
                "expected empty List in {}, got non-list value",
                column.name()
            );
        };
        assert!(
            empty.is_empty(),
            "{} retained values across row 1",
            column.name()
        );
    }

    assert_eq!(
        list_i32s(df.column("nested.field_0.field_0").unwrap().get(2).unwrap()),
        vec![Some(7)]
    );
    assert_eq!(
        list_strings(df.column("nested.field_0.field_1").unwrap().get(2).unwrap()),
        vec![Some("v3".to_owned())]
    );
    assert_eq!(
        list_bools(df.column("nested.field_1").unwrap().get(2).unwrap()),
        vec![Some(true)]
    );
}

#[test]
fn ancestor_and_nested_tuple_wrappers_compose() {
    let rows = [
        ComposedWrappers { nested: None },
        ComposedWrappers {
            nested: Some(Vec::new()),
        },
        ComposedWrappers {
            nested: Some(vec![
                (
                    Some((10, "c1".to_owned())),
                    true,
                    vec![(100, "x".to_owned()), (101, "y".to_owned())],
                ),
                (None, false, Vec::new()),
                (
                    Some((11, "c2".to_owned())),
                    true,
                    vec![(102, "z".to_owned())],
                ),
            ]),
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(df.shape(), (3, 5));
    assert_eq!(
        df.get_column_names(),
        [
            "nested.field_0.field_0",
            "nested.field_0.field_1",
            "nested.field_1",
            "nested.field_2.field_0",
            "nested.field_2.field_1",
        ]
    );

    for column in df.columns() {
        assert!(
            matches!(column.get(0).unwrap(), AnyValue::Null),
            "{} must be null when the ancestor Option is None",
            column.name()
        );
        let AnyValue::List(empty) = column.get(1).unwrap() else {
            panic!(
                "expected empty List in {}, got non-list value",
                column.name()
            );
        };
        assert!(
            empty.is_empty(),
            "{} retained values across the empty ancestor Vec",
            column.name()
        );
    }

    assert_eq!(
        list_i32s(df.column("nested.field_0.field_0").unwrap().get(2).unwrap()),
        vec![Some(10), None, Some(11)]
    );
    assert_eq!(
        list_strings(df.column("nested.field_0.field_1").unwrap().get(2).unwrap()),
        vec![Some("c1".to_owned()), None, Some("c2".to_owned())]
    );
    assert_eq!(
        list_bools(df.column("nested.field_1").unwrap().get(2).unwrap()),
        vec![Some(true), Some(false), Some(true)]
    );
    assert_eq!(
        nested_i32_lists(df.column("nested.field_2.field_0").unwrap().get(2).unwrap()),
        vec![
            Some(vec![Some(100), Some(101)]),
            Some(Vec::new()),
            Some(vec![Some(102)]),
        ]
    );
    assert_eq!(
        nested_string_lists(df.column("nested.field_2.field_1").unwrap().get(2).unwrap()),
        vec![
            Some(vec![Some("x".to_owned()), Some("y".to_owned())]),
            Some(Vec::new()),
            Some(vec![Some("z".to_owned())]),
        ]
    );
}

#[test]
fn all_absent_nested_payloads_preserve_nonzero_tuple_prefixes() {
    let rows = [
        AllAbsentNestedPayloads {
            nested: vec![(None, true), (None, false)],
        },
        AllAbsentNestedPayloads { nested: Vec::new() },
        AllAbsentNestedPayloads {
            nested: vec![(None, true)],
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(df.shape(), (3, 3));
    assert_eq!(
        df.get_column_names(),
        [
            "nested.field_0.id",
            "nested.field_0.label",
            "nested.field_1",
        ]
    );
    assert_eq!(
        list_i32s(df.column("nested.field_0.id").unwrap().get(0).unwrap()),
        vec![None, None]
    );
    assert_eq!(
        list_strings(df.column("nested.field_0.label").unwrap().get(0).unwrap()),
        vec![None, None]
    );
    assert_eq!(
        list_bools(df.column("nested.field_1").unwrap().get(0).unwrap()),
        vec![Some(true), Some(false)]
    );

    for column in df.columns() {
        let AnyValue::List(empty) = column.get(1).unwrap() else {
            panic!(
                "expected empty List in {}, got non-list value",
                column.name()
            );
        };
        assert!(empty.is_empty(), "{} must stay empty", column.name());
    }

    assert_eq!(
        list_i32s(df.column("nested.field_0.id").unwrap().get(2).unwrap()),
        vec![None]
    );
    assert_eq!(
        list_strings(df.column("nested.field_0.label").unwrap().get(2).unwrap()),
        vec![None]
    );
    assert_eq!(
        list_bools(df.column("nested.field_1").unwrap().get(2).unwrap()),
        vec![Some(true)]
    );
}
