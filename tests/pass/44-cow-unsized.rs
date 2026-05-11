use std::borrow::Cow;

use df_derive::ToDataFrame;
use polars::prelude::*;

#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct CowStrings<'a> {
    label: Cow<'a, str>,
    maybe_label: Option<Cow<'a, str>>,
    labels: Vec<Cow<'a, str>>,
    nullable_labels: Vec<Option<Cow<'a, str>>>,
}

#[derive(ToDataFrame, Clone)]
struct CowBytes<'a> {
    #[df_derive(as_binary)]
    payload: Cow<'a, [u8]>,
    #[df_derive(as_binary)]
    maybe_payload: Option<Cow<'a, [u8]>>,
    #[df_derive(as_binary)]
    payloads: Vec<Cow<'a, [u8]>>,
    #[df_derive(as_binary)]
    nullable_payloads: Vec<Option<Cow<'a, [u8]>>>,
    #[df_derive(as_binary)]
    maybe_payloads: Option<Vec<Cow<'a, [u8]>>>,
}

fn assert_string(df: &DataFrame, col: &str, row: usize, expected: &str) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::String(s) => assert_eq!(s, expected, "col {col} row {row}"),
        AnyValue::StringOwned(ref s) => assert_eq!(s.as_str(), expected, "col {col} row {row}"),
        other => panic!("unexpected AnyValue for {col} row {row}: {other:?}"),
    }
}

fn assert_binary(df: &DataFrame, col: &str, row: usize, expected: &[u8]) {
    match df.column(col).unwrap().get(row).unwrap() {
        AnyValue::Binary(b) => assert_eq!(b, expected, "col {col} row {row}"),
        AnyValue::BinaryOwned(ref b) => assert_eq!(b.as_slice(), expected, "col {col} row {row}"),
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
    let expected_owned: Vec<Option<Vec<u8>>> =
        expected.iter().map(|value| value.map(<[u8]>::to_vec)).collect();
    assert_eq!(actual, expected_owned, "col {col} row {row}");
}

fn main() {
    println!("--- Cow<'_, str> and Cow<'_, [u8]> support ---");

    let borrowed_strings = CowStrings {
        label: Cow::Borrowed("borrowed"),
        maybe_label: Some(Cow::Owned("maybe-owned".to_string())),
        labels: vec![Cow::Borrowed("alpha"), Cow::Owned("beta".to_string())],
        nullable_labels: vec![Some(Cow::Borrowed("present")), None],
    };
    let owned_strings = CowStrings {
        label: Cow::Owned("owned".to_string()),
        maybe_label: None,
        labels: vec![Cow::Owned("gamma".to_string())],
        nullable_labels: vec![Some(Cow::Owned("delta".to_string()))],
    };

    let string_schema = CowStrings::schema().unwrap();
    assert_eq!(schema_dtype(&string_schema, "label"), DataType::String);
    assert_eq!(schema_dtype(&string_schema, "maybe_label"), DataType::String);
    assert_eq!(
        schema_dtype(&string_schema, "labels"),
        DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        schema_dtype(&string_schema, "nullable_labels"),
        DataType::List(Box::new(DataType::String))
    );

    let string_df = borrowed_strings.clone().to_dataframe().unwrap();
    assert_string(&string_df, "label", 0, "borrowed");
    assert_string(&string_df, "maybe_label", 0, "maybe-owned");
    assert_string_list(&string_df, "labels", 0, &[Some("alpha"), Some("beta")]);
    assert_string_list(&string_df, "nullable_labels", 0, &[Some("present"), None]);

    let string_batch = vec![borrowed_strings, owned_strings];
    let string_batch_df = string_batch.as_slice().to_dataframe().unwrap();
    assert_string(&string_batch_df, "label", 1, "owned");
    assert_null(&string_batch_df, "maybe_label", 1);
    assert_string_list(&string_batch_df, "labels", 1, &[Some("gamma")]);
    assert_string_list(&string_batch_df, "nullable_labels", 1, &[Some("delta")]);

    let borrowed_bytes = CowBytes {
        payload: Cow::Borrowed(&b"borrowed"[..]),
        maybe_payload: Some(Cow::Owned(vec![1, 2, 3])),
        payloads: vec![Cow::Borrowed(&b"aa"[..]), Cow::Owned(vec![0, 255])],
        nullable_payloads: vec![Some(Cow::Borrowed(&b"present"[..])), None],
        maybe_payloads: Some(vec![Cow::Borrowed(&b"outer"[..]), Cow::Owned(vec![4, 5])]),
    };
    let owned_bytes = CowBytes {
        payload: Cow::Owned(vec![9, 8, 7]),
        maybe_payload: None,
        payloads: vec![Cow::Owned(vec![6])],
        nullable_payloads: vec![Some(Cow::Owned(vec![5, 4]))],
        maybe_payloads: None,
    };

    let bytes_schema = CowBytes::schema().unwrap();
    assert_eq!(schema_dtype(&bytes_schema, "payload"), DataType::Binary);
    assert_eq!(schema_dtype(&bytes_schema, "maybe_payload"), DataType::Binary);
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

    let bytes_df = borrowed_bytes.clone().to_dataframe().unwrap();
    assert_binary(&bytes_df, "payload", 0, b"borrowed");
    assert_binary(&bytes_df, "maybe_payload", 0, &[1, 2, 3]);
    assert_binary_list(&bytes_df, "payloads", 0, &[Some(b"aa"), Some(&[0, 255])]);
    assert_binary_list(
        &bytes_df,
        "nullable_payloads",
        0,
        &[Some(b"present"), None],
    );
    assert_binary_list(
        &bytes_df,
        "maybe_payloads",
        0,
        &[Some(b"outer"), Some(&[4, 5])],
    );

    let bytes_batch = vec![borrowed_bytes, owned_bytes];
    let bytes_batch_df = bytes_batch.as_slice().to_dataframe().unwrap();
    assert_binary(&bytes_batch_df, "payload", 1, &[9, 8, 7]);
    assert_null(&bytes_batch_df, "maybe_payload", 1);
    assert_binary_list(&bytes_batch_df, "payloads", 1, &[Some(&[6])]);
    assert_binary_list(&bytes_batch_df, "nullable_payloads", 1, &[Some(&[5, 4])]);
    assert_null(&bytes_batch_df, "maybe_payloads", 1);
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(name, _)| name == col)
        .map(|(_, dtype)| dtype.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}
