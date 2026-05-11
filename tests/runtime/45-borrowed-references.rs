use df_derive::ToDataFrame;
use polars::prelude::*;

use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct BorrowedRefs<'a> {
    label: &'a str,
    maybe_label: Option<&'a str>,
    labels: Vec<&'a str>,
    nullable_labels: Vec<Option<&'a str>>,
    owned_ref: &'a String,
    owned_refs: Vec<&'a String>,
    count: &'a i32,
    maybe_count: Option<&'a i32>,
    counts: Vec<&'a i32>,
    nullable_counts: Vec<Option<&'a i32>>,
}

#[derive(ToDataFrame, Clone)]
struct Inner<'a> {
    tag: &'a str,
    amount: i32,
}

#[derive(ToDataFrame, Clone)]
struct BorrowedNested<'a> {
    nested: &'a Inner<'a>,
    nested_vec: Vec<&'a Inner<'a>>,
}

#[derive(ToDataFrame, Clone)]
struct BorrowedBytes<'a> {
    #[df_derive(as_binary)]
    payload: &'a [u8],
    #[df_derive(as_binary)]
    maybe_payload: Option<&'a [u8]>,
    #[df_derive(as_binary)]
    payloads: Vec<&'a [u8]>,
    #[df_derive(as_binary)]
    nullable_payloads: Vec<Option<&'a [u8]>>,
    #[df_derive(as_binary)]
    maybe_payloads: Option<Vec<&'a [u8]>>,
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(name, _)| name == col)
        .map(|(_, dtype)| dtype.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}

fn assert_string(df: &DataFrame, col: &str, row: usize, expected: &str) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::String(s) => assert_eq!(s, expected, "col {col} row {row}"),
        AnyValue::StringOwned(ref s) => assert_eq!(s.as_str(), expected, "col {col} row {row}"),
        other => panic!("unexpected AnyValue for {col} row {row}: {other:?}"),
    }
}

fn assert_int(df: &DataFrame, col: &str, row: usize, expected: i32) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::Int32(v) => assert_eq!(v, expected, "col {col} row {row}"),
        other => panic!("unexpected AnyValue for {col} row {row}: {other:?}"),
    }
}

fn assert_null(df: &DataFrame, col: &str, row: usize) {
    let v = df.column(col).unwrap().get(row).unwrap();
    assert!(
        matches!(v, AnyValue::Null),
        "expected null at {col}[{row}], got {v:?}"
    );
}

fn assert_string_list(df: &DataFrame, col: &str, row: usize, expected: &[Option<&str>]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected List for {col} row {row}, got {v:?}");
    };
    let actual: Vec<Option<String>> = inner
        .iter()
        .map(|av| match av {
            AnyValue::String(s) => Some(s.to_string()),
            AnyValue::StringOwned(ref s) => Some(s.as_str().to_string()),
            AnyValue::Null => None,
            other => panic!("unexpected AnyValue inside list {col}: {other:?}"),
        })
        .collect();
    let expected_owned: Vec<Option<String>> = expected
        .iter()
        .map(|value| value.map(str::to_string))
        .collect();
    assert_eq!(actual, expected_owned, "col {col} row {row}");
}

fn assert_i32_list(df: &DataFrame, col: &str, row: usize, expected: &[Option<i32>]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected List for {col} row {row}, got {v:?}");
    };
    let actual: Vec<Option<i32>> = inner
        .iter()
        .map(|av| match av {
            AnyValue::Int32(v) => Some(v),
            AnyValue::Null => None,
            other => panic!("unexpected AnyValue inside list {col}: {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected, "col {col} row {row}");
}

fn assert_binary(df: &DataFrame, col: &str, row: usize, expected: &[u8]) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::Binary(b) => assert_eq!(b, expected, "col {col} row {row}"),
        AnyValue::BinaryOwned(ref b) => assert_eq!(b.as_slice(), expected, "col {col} row {row}"),
        other => panic!("unexpected AnyValue for {col} row {row}: {other:?}"),
    }
}

fn assert_binary_list(df: &DataFrame, col: &str, row: usize, expected: &[Option<&[u8]>]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected List for {col} row {row}, got {v:?}");
    };
    let actual: Vec<Option<Vec<u8>>> = inner
        .iter()
        .map(|av| match av {
            AnyValue::Binary(b) => Some(b.to_vec()),
            AnyValue::BinaryOwned(ref b) => Some(b.clone()),
            AnyValue::Null => None,
            other => panic!("unexpected AnyValue inside list {col}: {other:?}"),
        })
        .collect();
    let expected_owned: Vec<Option<Vec<u8>>> = expected
        .iter()
        .map(|value| value.map(<[u8]>::to_vec))
        .collect();
    assert_eq!(actual, expected_owned, "col {col} row {row}");
}

#[test]
fn runtime_semantics() {
    println!("--- Borrowed reference support ---");

    let owned = "owned-string".to_string();
    let owned_alt = "owned-alt".to_string();
    let a = 11;
    let b = 22;
    let refs = BorrowedRefs {
        label: "borrowed",
        maybe_label: Some("maybe"),
        labels: vec!["alpha", "beta"],
        nullable_labels: vec![Some("present"), None],
        owned_ref: &owned,
        owned_refs: vec![&owned, &owned_alt],
        count: &a,
        maybe_count: Some(&b),
        counts: vec![&a, &b],
        nullable_counts: vec![Some(&a), None],
    };

    let refs_schema = BorrowedRefs::schema().unwrap();
    assert_eq!(schema_dtype(&refs_schema, "label"), DataType::String);
    assert_eq!(schema_dtype(&refs_schema, "maybe_label"), DataType::String);
    assert_eq!(
        schema_dtype(&refs_schema, "labels"),
        DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        schema_dtype(&refs_schema, "nullable_labels"),
        DataType::List(Box::new(DataType::String))
    );
    assert_eq!(schema_dtype(&refs_schema, "owned_ref"), DataType::String);
    assert_eq!(
        schema_dtype(&refs_schema, "owned_refs"),
        DataType::List(Box::new(DataType::String))
    );
    assert_eq!(schema_dtype(&refs_schema, "count"), DataType::Int32);
    assert_eq!(schema_dtype(&refs_schema, "maybe_count"), DataType::Int32);
    assert_eq!(
        schema_dtype(&refs_schema, "counts"),
        DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(
        schema_dtype(&refs_schema, "nullable_counts"),
        DataType::List(Box::new(DataType::Int32))
    );

    let refs_df = refs.to_dataframe().unwrap();
    assert_string(&refs_df, "label", 0, "borrowed");
    assert_string(&refs_df, "maybe_label", 0, "maybe");
    assert_string_list(&refs_df, "labels", 0, &[Some("alpha"), Some("beta")]);
    assert_string_list(&refs_df, "nullable_labels", 0, &[Some("present"), None]);
    assert_string(&refs_df, "owned_ref", 0, "owned-string");
    assert_string_list(
        &refs_df,
        "owned_refs",
        0,
        &[Some("owned-string"), Some("owned-alt")],
    );
    assert_int(&refs_df, "count", 0, 11);
    assert_int(&refs_df, "maybe_count", 0, 22);
    assert_i32_list(&refs_df, "counts", 0, &[Some(11), Some(22)]);
    assert_i32_list(&refs_df, "nullable_counts", 0, &[Some(11), None]);

    let inner_a = Inner {
        tag: "inner-a",
        amount: 7,
    };
    let inner_b = Inner {
        tag: "inner-b",
        amount: 8,
    };
    let nested = BorrowedNested {
        nested: &inner_a,
        nested_vec: vec![&inner_a, &inner_b],
    };
    let nested_schema = BorrowedNested::schema().unwrap();
    assert_eq!(schema_dtype(&nested_schema, "nested.tag"), DataType::String);
    assert_eq!(
        schema_dtype(&nested_schema, "nested.amount"),
        DataType::Int32
    );
    assert_eq!(
        schema_dtype(&nested_schema, "nested_vec.tag"),
        DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        schema_dtype(&nested_schema, "nested_vec.amount"),
        DataType::List(Box::new(DataType::Int32))
    );
    let nested_df = nested.to_dataframe().unwrap();
    assert_string(&nested_df, "nested.tag", 0, "inner-a");
    assert_int(&nested_df, "nested.amount", 0, 7);
    assert_string_list(
        &nested_df,
        "nested_vec.tag",
        0,
        &[Some("inner-a"), Some("inner-b")],
    );
    assert_i32_list(&nested_df, "nested_vec.amount", 0, &[Some(7), Some(8)]);

    let bytes_a = BorrowedBytes {
        payload: &b"payload"[..],
        maybe_payload: Some(&b"maybe"[..]),
        payloads: vec![&b"aa"[..], &b"bb"[..]],
        nullable_payloads: vec![Some(&b"present"[..]), None],
        maybe_payloads: Some(vec![&b"outer-a"[..], &b"outer-b"[..]]),
    };
    let bytes_b = BorrowedBytes {
        payload: &b"payload-b"[..],
        maybe_payload: None,
        payloads: vec![&b"cc"[..]],
        nullable_payloads: vec![Some(&b"kept"[..])],
        maybe_payloads: None,
    };

    let bytes_schema = BorrowedBytes::schema().unwrap();
    assert_eq!(schema_dtype(&bytes_schema, "payload"), DataType::Binary);
    assert_eq!(
        schema_dtype(&bytes_schema, "maybe_payload"),
        DataType::Binary
    );
    assert_eq!(
        schema_dtype(&bytes_schema, "payloads"),
        DataType::List(Box::new(DataType::Binary))
    );
    assert_eq!(
        schema_dtype(&bytes_schema, "nullable_payloads"),
        DataType::List(Box::new(DataType::Binary))
    );
    assert_eq!(
        schema_dtype(&bytes_schema, "maybe_payloads"),
        DataType::List(Box::new(DataType::Binary))
    );

    let bytes_df = bytes_a.clone().to_dataframe().unwrap();
    assert_binary(&bytes_df, "payload", 0, b"payload");
    assert_binary(&bytes_df, "maybe_payload", 0, b"maybe");
    assert_binary_list(&bytes_df, "payloads", 0, &[Some(b"aa"), Some(b"bb")]);
    assert_binary_list(&bytes_df, "nullable_payloads", 0, &[Some(b"present"), None]);
    assert_binary_list(
        &bytes_df,
        "maybe_payloads",
        0,
        &[Some(b"outer-a"), Some(b"outer-b")],
    );

    let bytes_batch = vec![bytes_a, bytes_b];
    let bytes_batch_df = bytes_batch.as_slice().to_dataframe().unwrap();
    assert_binary(&bytes_batch_df, "payload", 1, b"payload-b");
    assert_null(&bytes_batch_df, "maybe_payload", 1);
    assert_binary_list(&bytes_batch_df, "payloads", 1, &[Some(b"cc")]);
    assert_binary_list(&bytes_batch_df, "nullable_payloads", 1, &[Some(b"kept")]);
    assert_null(&bytes_batch_df, "maybe_payloads", 1);
}
