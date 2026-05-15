use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;

#[derive(ToDataFrame, Clone)]
struct Inner {
    value: i32,
    label: String,
}

#[derive(ToDataFrame, Clone)]
struct Row {
    scalar: i32,
    opt_scalar: Option<i32>,
    nested: Inner,
    opt_nested: Option<Inner>,
    list: Vec<i32>,
    opt_list: Option<Vec<i32>>,
    list_opt: Vec<Option<i32>>,
    tuple: (i32, String),
    opt_tuple: Option<(i32, String)>,
    list_tuple: Vec<(i32, Option<String>)>,
    tuple_list: (Vec<i32>, Vec<Option<String>>),
}

fn row_zero() -> Row {
    Row {
        scalar: 1,
        opt_scalar: Some(2),
        nested: Inner {
            value: 10,
            label: "nested-0".to_owned(),
        },
        opt_nested: Some(Inner {
            value: 20,
            label: "opt-nested-0".to_owned(),
        }),
        list: vec![30, 31],
        opt_list: Some(vec![40, 41]),
        list_opt: vec![Some(50), None, Some(51)],
        tuple: (60, "tuple-0".to_owned()),
        opt_tuple: Some((70, "opt-tuple-0".to_owned())),
        list_tuple: vec![
            (80, Some("list-tuple-0a".to_owned())),
            (81, None),
            (82, Some("list-tuple-0b".to_owned())),
        ],
        tuple_list: (vec![90, 91], vec![Some("tuple-list-0a".to_owned()), None]),
    }
}

fn row_one() -> Row {
    Row {
        scalar: -1,
        opt_scalar: None,
        nested: Inner {
            value: 11,
            label: "nested-1".to_owned(),
        },
        opt_nested: None,
        list: Vec::new(),
        opt_list: None,
        list_opt: vec![None],
        tuple: (61, "tuple-1".to_owned()),
        opt_tuple: None,
        list_tuple: Vec::new(),
        tuple_list: (Vec::new(), Vec::new()),
    }
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(name, _)| name == col)
        .map(|(_, dtype)| dtype.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}

fn dataframe_schema(df: &DataFrame) -> Vec<(String, DataType)> {
    df.schema()
        .iter()
        .map(|(name, dtype)| (name.to_string(), dtype.clone()))
        .collect()
}

fn assert_schema(schema: &[(String, DataType)]) {
    let expected = [
        ("scalar", DataType::Int32),
        ("opt_scalar", DataType::Int32),
        ("nested.value", DataType::Int32),
        ("nested.label", DataType::String),
        ("opt_nested.value", DataType::Int32),
        ("opt_nested.label", DataType::String),
        ("list", DataType::List(Box::new(DataType::Int32))),
        ("opt_list", DataType::List(Box::new(DataType::Int32))),
        ("list_opt", DataType::List(Box::new(DataType::Int32))),
        ("tuple.field_0", DataType::Int32),
        ("tuple.field_1", DataType::String),
        ("opt_tuple.field_0", DataType::Int32),
        ("opt_tuple.field_1", DataType::String),
        (
            "list_tuple.field_0",
            DataType::List(Box::new(DataType::Int32)),
        ),
        (
            "list_tuple.field_1",
            DataType::List(Box::new(DataType::String)),
        ),
        (
            "tuple_list.field_0",
            DataType::List(Box::new(DataType::Int32)),
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

fn assert_i32(df: &DataFrame, col: &str, row: usize, expected: i32) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::Int32(value) => assert_eq!(value, expected, "col {col} row {row}"),
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

fn assert_i32_list(df: &DataFrame, col: &str, row: usize, expected: &[Option<i32>]) {
    let value = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = value else {
        panic!("expected List for {col} row {row}, got {value:?}");
    };
    let actual: Vec<Option<i32>> = inner.i32().unwrap().into_iter().collect();
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
    let expected: Vec<Option<String>> = expected
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect();
    assert_eq!(actual, expected, "col {col} row {row}");
}

#[test]
fn mixed_wrapper_shape_row_schema_and_values() {
    let schema = Row::schema().unwrap();
    assert_schema(&schema);

    let empty = Row::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 17));
    assert_schema(&dataframe_schema(&empty));

    let rows = vec![row_zero(), row_one()];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (2, 17));
    assert_schema(&dataframe_schema(&df));

    assert_i32(&df, "scalar", 0, 1);
    assert_i32(&df, "opt_scalar", 0, 2);
    assert_i32(&df, "nested.value", 0, 10);
    assert_string(&df, "nested.label", 0, "nested-0");
    assert_i32(&df, "opt_nested.value", 0, 20);
    assert_string(&df, "opt_nested.label", 0, "opt-nested-0");
    assert_i32_list(&df, "list", 0, &[Some(30), Some(31)]);
    assert_i32_list(&df, "opt_list", 0, &[Some(40), Some(41)]);
    assert_i32_list(&df, "list_opt", 0, &[Some(50), None, Some(51)]);
    assert_i32(&df, "tuple.field_0", 0, 60);
    assert_string(&df, "tuple.field_1", 0, "tuple-0");
    assert_i32(&df, "opt_tuple.field_0", 0, 70);
    assert_string(&df, "opt_tuple.field_1", 0, "opt-tuple-0");
    assert_i32_list(
        &df,
        "list_tuple.field_0",
        0,
        &[Some(80), Some(81), Some(82)],
    );
    assert_string_list(
        &df,
        "list_tuple.field_1",
        0,
        &[Some("list-tuple-0a"), None, Some("list-tuple-0b")],
    );
    assert_i32_list(&df, "tuple_list.field_0", 0, &[Some(90), Some(91)]);
    assert_string_list(&df, "tuple_list.field_1", 0, &[Some("tuple-list-0a"), None]);

    assert_i32(&df, "scalar", 1, -1);
    assert_null(&df, "opt_scalar", 1);
    assert_i32(&df, "nested.value", 1, 11);
    assert_string(&df, "nested.label", 1, "nested-1");
    assert_null(&df, "opt_nested.value", 1);
    assert_null(&df, "opt_nested.label", 1);
    assert_i32_list(&df, "list", 1, &[]);
    assert_null(&df, "opt_list", 1);
    assert_i32_list(&df, "list_opt", 1, &[None]);
    assert_i32(&df, "tuple.field_0", 1, 61);
    assert_string(&df, "tuple.field_1", 1, "tuple-1");
    assert_null(&df, "opt_tuple.field_0", 1);
    assert_null(&df, "opt_tuple.field_1", 1);
    assert_i32_list(&df, "list_tuple.field_0", 1, &[]);
    assert_string_list(&df, "list_tuple.field_1", 1, &[]);
    assert_i32_list(&df, "tuple_list.field_0", 1, &[]);
    assert_string_list(&df, "tuple_list.field_1", 1, &[]);
}
