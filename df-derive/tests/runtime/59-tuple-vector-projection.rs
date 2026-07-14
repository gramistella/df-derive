use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};
use polars::prelude::*;

#[derive(ToDataFrame, Clone)]
struct Inner {
    id: i32,
    label: String,
}

#[derive(ToDataFrame, Clone)]
#[allow(clippy::type_complexity)]
struct Row {
    a: Vec<(u32, String)>,
    b: Vec<(Vec<u32>, Option<String>)>,
    c: Option<(Vec<u32>, String)>,
    d: Vec<Option<(Vec<u32>, String)>>,
    e: Vec<(Option<Vec<u32>>, Option<String>)>,
    f: Vec<(Inner, Option<Inner>)>,
    g: Vec<Option<Option<(Vec<u32>, Option<String>)>>>,
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
        g: vec![
            Some(Some((vec![60, 61], Some("g1".to_owned())))),
            Some(None),
            None,
            Some(Some((Vec::new(), None))),
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
        g: Vec::new(),
    }
}

fn row_two() -> Row {
    Row {
        a: vec![(3, "a3".to_owned())],
        b: vec![
            (vec![12], Some("b2".to_owned())),
            (vec![13, 14, 15], Some("b3".to_owned())),
            (Vec::new(), None),
        ],
        c: Some((Vec::new(), "c2".to_owned())),
        d: vec![None, Some((vec![31, 32], "d2".to_owned()))],
        e: vec![(Some(vec![42]), None)],
        f: vec![
            (
                Inner {
                    id: 53,
                    label: "f3".to_owned(),
                },
                Some(Inner {
                    id: 54,
                    label: "f4".to_owned(),
                }),
            ),
            (
                Inner {
                    id: 55,
                    label: "f5".to_owned(),
                },
                None,
            ),
            (
                Inner {
                    id: 56,
                    label: "f6".to_owned(),
                },
                Some(Inner {
                    id: 57,
                    label: "f7".to_owned(),
                }),
            ),
        ],
        g: vec![Some(None), Some(Some((vec![62], Some("g2".to_owned()))))],
    }
}

fn schema_dtype(schema: &Schema, col: &str) -> DataType {
    schema
        .get(col)
        .cloned()
        .unwrap_or_else(|| panic!("column {col} missing"))
}

fn assert_schema(schema: &Schema) {
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
        ("g.field_0", list_list_u32()),
        ("g.field_1", list_string()),
    ];

    assert_eq!(schema.len(), expected.len());
    for (name, dtype) in expected {
        assert_eq!(schema_dtype(schema, name), dtype, "dtype for {name}");
    }
}

fn u32_list(value: AnyValue<'_>) -> Vec<Option<u32>> {
    match value {
        AnyValue::List(series) => series.u32().unwrap().iter().collect(),
        other => panic!("expected u32 List, got {other:?}"),
    }
}

fn i32_list(value: AnyValue<'_>) -> Vec<Option<i32>> {
    match value {
        AnyValue::List(series) => series.i32().unwrap().iter().collect(),
        other => panic!("expected i32 List, got {other:?}"),
    }
}

fn string_list(value: AnyValue<'_>) -> Vec<Option<String>> {
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

fn nested_u32_lists(value: AnyValue<'_>) -> Vec<Option<Vec<Option<u32>>>> {
    let AnyValue::List(outer) = value else {
        panic!("expected outer List, got {value:?}");
    };
    (0..outer.len())
        .map(|idx| match outer.get(idx).unwrap() {
            AnyValue::List(inner) => Some(inner.u32().unwrap().iter().collect()),
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

fn list_width(df: &DataFrame, col: &str, row: usize) -> usize {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::List(series) => series.len(),
        other => panic!("expected List at {col}[{row}], got {other:?}"),
    }
}

fn assert_tuple_sibling_widths(df: &DataFrame) {
    let groups: &[&[&str]] = &[
        &["a.field_0", "a.field_1"],
        &["b.field_0", "b.field_1"],
        &["d.field_0", "d.field_1"],
        &["e.field_0", "e.field_1"],
        &[
            "f.field_0.id",
            "f.field_0.label",
            "f.field_1.id",
            "f.field_1.label",
        ],
        &["g.field_0", "g.field_1"],
    ];

    for row in 0..df.height() {
        for columns in groups {
            let expected = list_width(df, columns[0], row);
            for column in &columns[1..] {
                assert_eq!(
                    list_width(df, column, row),
                    expected,
                    "tuple siblings diverged at row {row}: {columns:?}"
                );
            }
        }
    }
}

fn assert_row_after_empty_boundary(df: &DataFrame) {
    assert_eq!(
        u32_list(df.column("a.field_0").unwrap().get(2).unwrap()),
        vec![Some(3)]
    );
    assert_eq!(
        string_list(df.column("a.field_1").unwrap().get(2).unwrap()),
        vec![Some("a3".to_owned())]
    );
    assert_eq!(
        nested_u32_lists(df.column("b.field_0").unwrap().get(2).unwrap()),
        vec![
            Some(vec![Some(12)]),
            Some(vec![Some(13), Some(14), Some(15)]),
            Some(Vec::new()),
        ]
    );
    assert_eq!(
        string_list(df.column("b.field_1").unwrap().get(2).unwrap()),
        vec![Some("b2".to_owned()), Some("b3".to_owned()), None]
    );
    assert_eq!(
        u32_list(df.column("c.field_0").unwrap().get(2).unwrap()),
        Vec::<Option<u32>>::new()
    );
    assert_eq!(
        df.column("c.field_1").unwrap().get(2).unwrap(),
        AnyValue::String("c2")
    );
    assert_eq!(
        nested_u32_lists(df.column("d.field_0").unwrap().get(2).unwrap()),
        vec![None, Some(vec![Some(31), Some(32)])]
    );
    assert_eq!(
        string_list(df.column("d.field_1").unwrap().get(2).unwrap()),
        vec![None, Some("d2".to_owned())]
    );
    assert_eq!(
        nested_u32_lists(df.column("e.field_0").unwrap().get(2).unwrap()),
        vec![Some(vec![Some(42)])]
    );
    assert_eq!(
        string_list(df.column("e.field_1").unwrap().get(2).unwrap()),
        vec![None]
    );
    assert_eq!(
        i32_list(df.column("f.field_0.id").unwrap().get(2).unwrap()),
        vec![Some(53), Some(55), Some(56)]
    );
    assert_eq!(
        i32_list(df.column("f.field_1.id").unwrap().get(2).unwrap()),
        vec![Some(54), None, Some(57)]
    );
    assert_eq!(
        nested_u32_lists(df.column("g.field_0").unwrap().get(2).unwrap()),
        vec![None, Some(vec![Some(62)])]
    );
    assert_eq!(
        string_list(df.column("g.field_1").unwrap().get(2).unwrap()),
        vec![None, Some("g2".to_owned())]
    );
}

#[test]
fn tuple_vector_projection_schema_and_values() {
    let schema = Row::schema().unwrap();
    assert_schema(&schema);

    let empty = Row::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 16));
    assert_schema(empty.schema());

    let rows = vec![row_zero(), row_one(), row_two()];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (3, 16));
    assert_schema(df.schema());
    assert_tuple_sibling_widths(&df);

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
        nested_u32_lists(df.column("g.field_0").unwrap().get(0).unwrap()),
        vec![Some(vec![Some(60), Some(61)]), None, None, Some(Vec::new()),]
    );
    assert_eq!(
        string_list(df.column("g.field_1").unwrap().get(0).unwrap()),
        vec![Some("g1".to_owned()), None, None, None]
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
    assert_eq!(
        nested_u32_lists(df.column("g.field_0").unwrap().get(1).unwrap()),
        Vec::<Option<Vec<Option<u32>>>>::new()
    );
    assert_row_after_empty_boundary(&df);
}
