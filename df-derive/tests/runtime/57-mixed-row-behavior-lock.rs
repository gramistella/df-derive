use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;

#[derive(ToDataFrame, Clone)]
struct Inner {
    inner_id: u64,
    label: String,
}

#[derive(ToDataFrame, Clone)]
struct Row {
    id: u64,
    maybe_id: Option<u64>,
    ids: Vec<u64>,
    maybe_ids: Option<Vec<u64>>,
    ids_with_nulls: Vec<Option<u64>>,
    nested: Inner,
    maybe_nested: Option<Inner>,
    nested_list: Vec<Inner>,
    tuple: (u64, String),
    tuple_list: Vec<(u64, Option<String>)>,
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(name, _)| name == col)
        .map(|(_, dtype)| dtype.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}

fn assert_schema(schema: &[(String, DataType)]) {
    let expected = [
        ("id", DataType::UInt64),
        ("maybe_id", DataType::UInt64),
        ("ids", DataType::List(Box::new(DataType::UInt64))),
        ("maybe_ids", DataType::List(Box::new(DataType::UInt64))),
        ("ids_with_nulls", DataType::List(Box::new(DataType::UInt64))),
        ("nested.inner_id", DataType::UInt64),
        ("nested.label", DataType::String),
        ("maybe_nested.inner_id", DataType::UInt64),
        ("maybe_nested.label", DataType::String),
        (
            "nested_list.inner_id",
            DataType::List(Box::new(DataType::UInt64)),
        ),
        (
            "nested_list.label",
            DataType::List(Box::new(DataType::String)),
        ),
        ("tuple.field_0", DataType::UInt64),
        ("tuple.field_1", DataType::String),
        (
            "tuple_list.field_0",
            DataType::List(Box::new(DataType::UInt64)),
        ),
        (
            "tuple_list.field_1",
            DataType::List(Box::new(DataType::String)),
        ),
    ];

    assert_eq!(schema.len(), expected.len());
    for (name, dtype) in expected {
        assert_eq!(schema_dtype(schema, name), dtype, "dtype for {name}");
    }
}

fn assert_u64(df: &DataFrame, col: &str, row: usize, expected: u64) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::UInt64(value) => assert_eq!(value, expected, "col {col} row {row}"),
        other => panic!("unexpected AnyValue for {col} row {row}: {other:?}"),
    }
}

fn assert_string(df: &DataFrame, col: &str, row: usize, expected: &str) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::String(value) => assert_eq!(value, expected, "col {col} row {row}"),
        AnyValue::StringOwned(ref value) => {
            assert_eq!(value.as_str(), expected, "col {col} row {row}");
        }
        other => panic!("unexpected AnyValue for {col} row {row}: {other:?}"),
    }
}

fn assert_null(df: &DataFrame, col: &str, row: usize) {
    let value = df.column(col).unwrap().get(row).unwrap();
    assert!(
        matches!(value, AnyValue::Null),
        "expected null at {col}[{row}], got {value:?}"
    );
}

fn assert_u64_list(df: &DataFrame, col: &str, row: usize, expected: &[Option<u64>]) {
    let value = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = value else {
        panic!("expected List for {col} row {row}, got {value:?}");
    };
    let actual: Vec<Option<u64>> = inner.u64().unwrap().into_iter().collect();
    assert_eq!(actual, expected, "col {col} row {row}");
}

fn assert_string_list(df: &DataFrame, col: &str, row: usize, expected: &[Option<&str>]) {
    let value = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = value else {
        panic!("expected List for {col} row {row}, got {value:?}");
    };
    let actual: Vec<Option<String>> = inner
        .str()
        .unwrap()
        .into_iter()
        .map(|value| value.map(str::to_owned))
        .collect();
    let expected_owned: Vec<Option<String>> = expected
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect();
    assert_eq!(actual, expected_owned, "col {col} row {row}");
}

fn row_zero() -> Row {
    Row {
        id: 1,
        maybe_id: Some(2),
        ids: vec![10, 11],
        maybe_ids: Some(vec![20, 21]),
        ids_with_nulls: vec![Some(30), None, Some(31)],
        nested: Inner {
            inner_id: 100,
            label: "nested-0".to_owned(),
        },
        maybe_nested: Some(Inner {
            inner_id: 200,
            label: "maybe-nested-0".to_owned(),
        }),
        nested_list: vec![
            Inner {
                inner_id: 300,
                label: "list-0a".to_owned(),
            },
            Inner {
                inner_id: 301,
                label: "list-0b".to_owned(),
            },
        ],
        tuple: (400, "tuple-0".to_owned()),
        tuple_list: vec![(500, Some("tuple-list-0a".to_owned())), (501, None)],
    }
}

fn row_one() -> Row {
    Row {
        id: 3,
        maybe_id: None,
        ids: Vec::new(),
        maybe_ids: None,
        ids_with_nulls: vec![None],
        nested: Inner {
            inner_id: 101,
            label: "nested-1".to_owned(),
        },
        maybe_nested: None,
        nested_list: Vec::new(),
        tuple: (401, "tuple-1".to_owned()),
        tuple_list: Vec::new(),
    }
}

fn row_two() -> Row {
    Row {
        id: 4,
        maybe_id: Some(5),
        ids: vec![12],
        maybe_ids: Some(Vec::new()),
        ids_with_nulls: vec![Some(32)],
        nested: Inner {
            inner_id: 102,
            label: "nested-2".to_owned(),
        },
        maybe_nested: Some(Inner {
            inner_id: 202,
            label: "maybe-nested-2".to_owned(),
        }),
        nested_list: vec![Inner {
            inner_id: 302,
            label: "list-2a".to_owned(),
        }],
        tuple: (402, "tuple-2".to_owned()),
        tuple_list: vec![(502, Some("tuple-list-2a".to_owned()))],
    }
}

#[test]
fn mixed_row_shapes_keep_schema_and_values() {
    let schema = Row::schema().unwrap();
    assert_schema(&schema);

    let empty = Row::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 15));
    assert_schema(
        &empty
            .schema()
            .iter()
            .map(|(name, dtype)| (name.to_string(), dtype.clone()))
            .collect::<Vec<_>>(),
    );

    let single = row_zero().to_dataframe().unwrap();
    assert_eq!(single.shape(), (1, 15));
    assert_schema(
        &single
            .schema()
            .iter()
            .map(|(name, dtype)| (name.to_string(), dtype.clone()))
            .collect::<Vec<_>>(),
    );
    assert_u64(&single, "id", 0, 1);
    assert_u64(&single, "maybe_id", 0, 2);
    assert_u64_list(&single, "ids", 0, &[Some(10), Some(11)]);
    assert_u64_list(&single, "maybe_ids", 0, &[Some(20), Some(21)]);
    assert_u64_list(&single, "ids_with_nulls", 0, &[Some(30), None, Some(31)]);
    assert_u64(&single, "nested.inner_id", 0, 100);
    assert_string(&single, "nested.label", 0, "nested-0");
    assert_u64(&single, "maybe_nested.inner_id", 0, 200);
    assert_string(&single, "maybe_nested.label", 0, "maybe-nested-0");
    assert_u64_list(&single, "nested_list.inner_id", 0, &[Some(300), Some(301)]);
    assert_string_list(
        &single,
        "nested_list.label",
        0,
        &[Some("list-0a"), Some("list-0b")],
    );
    assert_u64(&single, "tuple.field_0", 0, 400);
    assert_string(&single, "tuple.field_1", 0, "tuple-0");
    assert_u64_list(&single, "tuple_list.field_0", 0, &[Some(500), Some(501)]);
    assert_string_list(
        &single,
        "tuple_list.field_1",
        0,
        &[Some("tuple-list-0a"), None],
    );

    let rows = vec![row_zero(), row_one(), row_two()];
    let batch = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape(), (3, 15));

    assert_u64(&batch, "id", 1, 3);
    assert_null(&batch, "maybe_id", 1);
    assert_u64_list(&batch, "ids", 1, &[]);
    assert_null(&batch, "maybe_ids", 1);
    assert_u64_list(&batch, "ids_with_nulls", 1, &[None]);
    assert_u64(&batch, "nested.inner_id", 1, 101);
    assert_string(&batch, "nested.label", 1, "nested-1");
    assert_null(&batch, "maybe_nested.inner_id", 1);
    assert_null(&batch, "maybe_nested.label", 1);
    assert_u64_list(&batch, "nested_list.inner_id", 1, &[]);
    assert_string_list(&batch, "nested_list.label", 1, &[]);
    assert_u64(&batch, "tuple.field_0", 1, 401);
    assert_string(&batch, "tuple.field_1", 1, "tuple-1");
    assert_u64_list(&batch, "tuple_list.field_0", 1, &[]);
    assert_string_list(&batch, "tuple_list.field_1", 1, &[]);

    assert_u64(&batch, "maybe_id", 2, 5);
    assert_u64_list(&batch, "maybe_ids", 2, &[]);
    assert_u64_list(&batch, "nested_list.inner_id", 2, &[Some(302)]);
    assert_string_list(&batch, "nested_list.label", 2, &[Some("list-2a")]);
    assert_u64_list(&batch, "tuple_list.field_0", 2, &[Some(502)]);
    assert_string_list(&batch, "tuple_list.field_1", 2, &[Some("tuple-list-2a")]);
}
