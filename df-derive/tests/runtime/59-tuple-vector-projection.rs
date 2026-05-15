use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;

#[derive(ToDataFrame, Clone)]
struct Inner {
    id: i32,
    label: String,
}

#[derive(ToDataFrame, Clone)]
struct Row {
    a: Vec<(u32, String)>,
    b: Vec<(Vec<u32>, Option<String>)>,
    c: Option<(Vec<u32>, String)>,
    d: Vec<Option<(Vec<u32>, String)>>,
    e: Vec<(Option<Vec<u32>>, Option<String>)>,
    f: Vec<(Inner, Option<Inner>)>,
}

fn row_zero() -> Row {
    Row {
        a: vec![(1, "a1".to_owned()), (2, "a2".to_owned())],
        b: vec![(vec![10, 11], Some("b1".to_owned())), (Vec::new(), None)],
        c: Some((vec![20, 21], "c1".to_owned())),
        d: vec![
            Some((vec![30], "d1".to_owned())),
            None,
            Some((Vec::new(), "d3".to_owned())),
        ],
        e: vec![
            (Some(vec![40, 41]), Some("e1".to_owned())),
            (None, None),
            (Some(Vec::new()), Some("e3".to_owned())),
        ],
        f: vec![
            (
                Inner {
                    id: 50,
                    label: "f0".to_owned(),
                },
                Some(Inner {
                    id: 51,
                    label: "f1".to_owned(),
                }),
            ),
            (
                Inner {
                    id: 52,
                    label: "f2".to_owned(),
                },
                None,
            ),
        ],
    }
}

fn row_one() -> Row {
    Row {
        a: Vec::new(),
        b: Vec::new(),
        c: None,
        d: Vec::new(),
        e: Vec::new(),
        f: Vec::new(),
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
    let list_u32 = || DataType::List(Box::new(DataType::UInt32));
    let list_string = || DataType::List(Box::new(DataType::String));
    let list_list_u32 = || DataType::List(Box::new(list_u32()));
    let expected = [
        ("a.field_0", list_u32()),
        ("a.field_1", list_string()),
        ("b.field_0", list_list_u32()),
        ("b.field_1", list_string()),
        ("c.field_0", list_u32()),
        ("c.field_1", DataType::String),
        ("d.field_0", list_list_u32()),
        ("d.field_1", list_string()),
        ("e.field_0", list_list_u32()),
        ("e.field_1", list_string()),
        ("f.field_0.id", DataType::List(Box::new(DataType::Int32))),
        ("f.field_0.label", list_string()),
        ("f.field_1.id", DataType::List(Box::new(DataType::Int32))),
        ("f.field_1.label", list_string()),
    ];

    assert_eq!(schema.len(), expected.len());
    for (name, dtype) in expected {
        assert_eq!(schema_dtype(schema, name), dtype, "dtype for {name}");
    }
}

fn u32_list(value: AnyValue<'_>) -> Vec<Option<u32>> {
    match value {
        AnyValue::List(series) => series.u32().unwrap().into_iter().collect(),
        other => panic!("expected u32 List, got {other:?}"),
    }
}

fn i32_list(value: AnyValue<'_>) -> Vec<Option<i32>> {
    match value {
        AnyValue::List(series) => series.i32().unwrap().into_iter().collect(),
        other => panic!("expected i32 List, got {other:?}"),
    }
}

fn string_list(value: AnyValue<'_>) -> Vec<Option<String>> {
    match value {
        AnyValue::List(series) => series
            .str()
            .unwrap()
            .into_iter()
            .map(|value| value.map(str::to_owned))
            .collect(),
        other => panic!("expected string List, got {other:?}"),
    }
}

fn nested_u32_lists(value: AnyValue<'_>) -> Vec<Option<Vec<Option<u32>>>> {
    let AnyValue::List(outer) = value else {
        panic!("expected outer List, got {value:?}");
    };
    (0..outer.len())
        .map(|idx| match outer.get(idx).unwrap() {
            AnyValue::List(inner) => Some(inner.u32().unwrap().into_iter().collect()),
            AnyValue::Null => None,
            other => panic!("expected inner u32 List or Null, got {other:?}"),
        })
        .collect()
}

fn assert_null(df: &DataFrame, col: &str, row: usize) {
    let value = df.column(col).unwrap().get(row).unwrap();
    assert!(
        matches!(value, AnyValue::Null),
        "expected null at {col}[{row}], got {value:?}"
    );
}

#[test]
fn tuple_vector_projection_schema_and_values() {
    let schema = Row::schema().unwrap();
    assert_schema(&schema);

    let empty = Row::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 14));
    assert_schema(&dataframe_schema(&empty));

    let rows = vec![row_zero(), row_one()];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (2, 14));
    assert_schema(&dataframe_schema(&df));

    assert_eq!(
        u32_list(df.column("a.field_0").unwrap().get(0).unwrap()),
        vec![Some(1), Some(2)]
    );
    assert_eq!(
        string_list(df.column("a.field_1").unwrap().get(0).unwrap()),
        vec![Some("a1".to_owned()), Some("a2".to_owned())]
    );
    assert_eq!(
        nested_u32_lists(df.column("b.field_0").unwrap().get(0).unwrap()),
        vec![Some(vec![Some(10), Some(11)]), Some(Vec::new())]
    );
    assert_eq!(
        string_list(df.column("b.field_1").unwrap().get(0).unwrap()),
        vec![Some("b1".to_owned()), None]
    );
    assert_eq!(
        u32_list(df.column("c.field_0").unwrap().get(0).unwrap()),
        vec![Some(20), Some(21)]
    );
    assert_eq!(
        df.column("c.field_1").unwrap().get(0).unwrap(),
        AnyValue::String("c1")
    );
    assert_eq!(
        nested_u32_lists(df.column("d.field_0").unwrap().get(0).unwrap()),
        vec![Some(vec![Some(30)]), None, Some(Vec::new())]
    );
    assert_eq!(
        string_list(df.column("d.field_1").unwrap().get(0).unwrap()),
        vec![Some("d1".to_owned()), None, Some("d3".to_owned())]
    );
    assert_eq!(
        nested_u32_lists(df.column("e.field_0").unwrap().get(0).unwrap()),
        vec![Some(vec![Some(40), Some(41)]), None, Some(Vec::new())]
    );
    assert_eq!(
        string_list(df.column("e.field_1").unwrap().get(0).unwrap()),
        vec![Some("e1".to_owned()), None, Some("e3".to_owned())]
    );
    assert_eq!(
        i32_list(df.column("f.field_0.id").unwrap().get(0).unwrap()),
        vec![Some(50), Some(52)]
    );
    assert_eq!(
        string_list(df.column("f.field_0.label").unwrap().get(0).unwrap()),
        vec![Some("f0".to_owned()), Some("f2".to_owned())]
    );
    assert_eq!(
        i32_list(df.column("f.field_1.id").unwrap().get(0).unwrap()),
        vec![Some(51), None]
    );
    assert_eq!(
        string_list(df.column("f.field_1.label").unwrap().get(0).unwrap()),
        vec![Some("f1".to_owned()), None]
    );

    assert_eq!(
        u32_list(df.column("a.field_0").unwrap().get(1).unwrap()),
        Vec::<Option<u32>>::new()
    );
    assert_eq!(
        nested_u32_lists(df.column("b.field_0").unwrap().get(1).unwrap()),
        Vec::<Option<Vec<Option<u32>>>>::new()
    );
    assert_null(&df, "c.field_0", 1);
    assert_null(&df, "c.field_1", 1);
    assert_eq!(
        nested_u32_lists(df.column("d.field_0").unwrap().get(1).unwrap()),
        Vec::<Option<Vec<Option<u32>>>>::new()
    );
    assert_eq!(
        string_list(df.column("f.field_1.label").unwrap().get(1).unwrap()),
        Vec::<Option<String>>::new()
    );
}
