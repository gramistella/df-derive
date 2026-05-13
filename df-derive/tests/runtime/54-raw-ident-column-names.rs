use crate::core::dataframe::ToDataFrame;
use df_derive::ToDataFrame;
use polars::prelude::AnyValue;

#[derive(ToDataFrame, Clone)]
struct Inner {
    r#type: i32,
}

#[derive(ToDataFrame, Clone)]
struct Row {
    r#type: String,
    r#struct: Inner,
    r#match: (bool, i32),
}

#[test]
fn raw_identifier_prefixes_do_not_leak_into_column_names() {
    let row = Row {
        r#type: "equity".to_owned(),
        r#struct: Inner { r#type: 42 },
        r#match: (true, 7),
    };

    let schema_names: Vec<String> = Row::schema()
        .unwrap()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        schema_names,
        ["type", "struct.type", "match.field_0", "match.field_1"]
    );

    let df = row.to_dataframe().unwrap();
    let column_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|name| name.as_str().to_owned())
        .collect();
    assert_eq!(
        column_names,
        ["type", "struct.type", "match.field_0", "match.field_1"]
    );
    assert_eq!(
        df.column("type").unwrap().get(0).unwrap(),
        AnyValue::String("equity")
    );
    assert_eq!(
        df.column("struct.type").unwrap().get(0).unwrap(),
        AnyValue::Int32(42)
    );
    assert_eq!(
        df.column("match.field_0").unwrap().get(0).unwrap(),
        AnyValue::Boolean(true)
    );
    assert_eq!(
        df.column("match.field_1").unwrap().get(0).unwrap(),
        AnyValue::Int32(7)
    );
}
