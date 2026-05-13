// Smart-pointer transparency: `Box<T>`, `Rc<T>`, `Arc<T>`, and `Cow<'a, T>`
// (with sized inner) peel transparently at parse time and produce the same
// column shape as the inner type. Unsized `Cow<'_, str>` / `Cow<'_, [u8]>`
// have their own semantic tests in 44-cow-unsized.rs; borrowed references
// are covered in 45-borrowed-references.rs.
//
// This test pins every supported composition shape plus the previously-
// blocked `Box<Vec<u8>>` + `as_binary` regression.
#![allow(clippy::box_collection, clippy::redundant_allocation, clippy::vec_box)]

use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;
// Cow import is used by the `cow_date: Cow<'static, NaiveDate>` field below.

use chrono::NaiveDate;
use df_derive::ToDataFrame;
use polars::prelude::*;

use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct Bare {
    bx_i64: Box<i64>,
    arc_str: Arc<String>,
    rc_date: Rc<NaiveDate>,
    bx_date: Box<NaiveDate>,
    cow_date: Cow<'static, NaiveDate>,
    bx_dur: Box<chrono::Duration>,
    bxbx_i32: Box<Box<i32>>,
}

#[derive(ToDataFrame, Clone)]
struct Composed {
    opt_bx_i32: Option<Box<i32>>,
    bx_opt_bool: Box<Option<bool>>,
    vec_arc_string: Vec<Arc<String>>,
    bx_vec_f64: Box<Vec<f64>>,
}

#[derive(ToDataFrame, Clone)]
struct BoundaryPositions {
    opt_box_vec: Option<Box<Vec<i32>>>,
    opt_box_opt: Option<Box<Option<i32>>>,
    vec_box_opt: Vec<Box<Option<i32>>>,
    vec_box_vec: Vec<Box<Vec<i32>>>,
    vec_opt_arc_vec: Vec<Option<Arc<Vec<i32>>>>,
    #[df_derive(as_binary)]
    opt_box_blob: Option<Box<Vec<u8>>>,
}

#[derive(ToDataFrame, Clone)]
struct Regression {
    #[df_derive(as_binary)]
    bx_blob: Box<Vec<u8>>,
    arc_date: Arc<NaiveDate>,
    bx_chrono_dur: Box<chrono::Duration>,
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(n, _)| n == col)
        .map(|(_, dt)| dt.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}

fn assert_list_i32(df: &DataFrame, col: &str, row: usize, expected: &[Option<i32>]) {
    let AnyValue::List(inner) = df.column(col).unwrap().get(row).unwrap() else {
        panic!("expected List for {col}[{row}]");
    };
    let actual: Vec<Option<i32>> = inner.i32().unwrap().into_iter().collect();
    assert_eq!(actual, expected, "{col}[{row}]");
}

fn assert_nested_list_i32(
    df: &DataFrame,
    col: &str,
    row: usize,
    expected: &[Option<Vec<Option<i32>>>],
) {
    let AnyValue::List(outer) = df.column(col).unwrap().get(row).unwrap() else {
        panic!("expected outer List for {col}[{row}]");
    };
    let actual: Vec<Option<Vec<Option<i32>>>> = outer
        .iter()
        .map(|value| match value {
            AnyValue::List(inner) => Some(inner.i32().unwrap().into_iter().collect()),
            AnyValue::Null => None,
            other => panic!("unexpected nested list value for {col}[{row}]: {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected, "{col}[{row}]");
}

#[test]
fn runtime_semantics() {
    println!("--- Smart-pointer transparency ---");

    // --- Bare composition table ---
    let bare_schema = Bare::schema().unwrap();
    assert_eq!(schema_dtype(&bare_schema, "bx_i64"), DataType::Int64);
    assert_eq!(schema_dtype(&bare_schema, "arc_str"), DataType::String);
    assert_eq!(schema_dtype(&bare_schema, "rc_date"), DataType::Date);
    assert_eq!(schema_dtype(&bare_schema, "bx_date"), DataType::Date);
    assert_eq!(schema_dtype(&bare_schema, "cow_date"), DataType::Date);
    assert_eq!(
        schema_dtype(&bare_schema, "bx_dur"),
        DataType::Duration(TimeUnit::Nanoseconds)
    );
    assert_eq!(schema_dtype(&bare_schema, "bxbx_i32"), DataType::Int32);

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let leap = NaiveDate::from_ymd_opt(2000, 2, 29).unwrap();
    let bare0 = Bare {
        bx_i64: Box::new(42),
        arc_str: Arc::new("hello".to_string()),
        rc_date: Rc::new(epoch),
        bx_date: Box::new(leap),
        cow_date: Cow::Owned(leap),
        bx_dur: Box::new(chrono::Duration::nanoseconds(1_500)),
        bxbx_i32: Box::new(Box::new(-7)),
    };
    let bare1 = bare0.clone();
    let bare_df = vec![bare0, bare1].as_slice().to_dataframe().unwrap();
    assert_eq!(bare_df.shape(), (2, 7));

    // Pin a few values via AnyValue extraction.
    assert!(matches!(
        bare_df.column("bx_i64").unwrap().get(0).unwrap(),
        AnyValue::Int64(42)
    ));
    assert!(matches!(
        bare_df.column("bxbx_i32").unwrap().get(0).unwrap(),
        AnyValue::Int32(-7)
    ));
    match bare_df.column("arc_str").unwrap().get(0).unwrap() {
        AnyValue::String(s) => assert_eq!(s, "hello"),
        AnyValue::StringOwned(ref s) => assert_eq!(s.as_str(), "hello"),
        other => panic!("unexpected arc_str AnyValue {other:?}"),
    }
    match bare_df.column("rc_date").unwrap().get(0).unwrap() {
        AnyValue::Date(d) => assert_eq!(d, 0),
        other => panic!("unexpected rc_date AnyValue {other:?}"),
    }
    match bare_df.column("bx_date").unwrap().get(0).unwrap() {
        AnyValue::Date(d) => {
            assert_eq!(d, leap.signed_duration_since(epoch).num_days() as i32)
        }
        other => panic!("unexpected bx_date AnyValue {other:?}"),
    }
    match bare_df.column("bx_dur").unwrap().get(0).unwrap() {
        AnyValue::Duration(v, u) => {
            assert_eq!(u, TimeUnit::Nanoseconds);
            assert_eq!(v, 1_500);
        }
        other => panic!("unexpected bx_dur AnyValue {other:?}"),
    }

    // --- Composed (option / vec compositions over smart pointers) ---
    let composed_schema = Composed::schema().unwrap();
    assert_eq!(
        schema_dtype(&composed_schema, "opt_bx_i32"),
        DataType::Int32
    );
    assert_eq!(
        schema_dtype(&composed_schema, "bx_opt_bool"),
        DataType::Boolean
    );
    assert_eq!(
        schema_dtype(&composed_schema, "vec_arc_string"),
        DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        schema_dtype(&composed_schema, "bx_vec_f64"),
        DataType::List(Box::new(DataType::Float64))
    );

    let row0 = Composed {
        opt_bx_i32: Some(Box::new(123)),
        bx_opt_bool: Box::new(Some(true)),
        vec_arc_string: vec![Arc::new("a".to_string()), Arc::new("b".to_string())],
        bx_vec_f64: Box::new(vec![1.5, 2.5, 3.5]),
    };
    let row1 = Composed {
        opt_bx_i32: None,
        bx_opt_bool: Box::new(None),
        vec_arc_string: vec![],
        bx_vec_f64: Box::new(vec![]),
    };
    let composed_df = vec![row0, row1].as_slice().to_dataframe().unwrap();
    assert_eq!(composed_df.shape(), (2, 4));

    // Option<Box<i32>> validity check
    assert!(matches!(
        composed_df.column("opt_bx_i32").unwrap().get(0).unwrap(),
        AnyValue::Int32(123)
    ));
    assert!(matches!(
        composed_df.column("opt_bx_i32").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));

    // Box<Option<bool>> validity check
    assert!(matches!(
        composed_df.column("bx_opt_bool").unwrap().get(0).unwrap(),
        AnyValue::Boolean(true)
    ));
    assert!(matches!(
        composed_df.column("bx_opt_bool").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));

    // Vec<Arc<String>> list extraction
    let v = composed_df
        .column("vec_arc_string")
        .unwrap()
        .get(0)
        .unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected list")
    };
    let strs: Vec<String> = inner
        .iter()
        .map(|av| match av {
            AnyValue::String(s) => s.to_string(),
            AnyValue::StringOwned(s) => s.to_string(),
            other => panic!("unexpected list elem {other:?}"),
        })
        .collect();
    assert_eq!(strs, vec!["a".to_string(), "b".to_string()]);

    // Box<Vec<f64>> list
    let v = composed_df.column("bx_vec_f64").unwrap().get(0).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected list")
    };
    let nums: Vec<f64> = inner
        .iter()
        .map(|av| match av {
            AnyValue::Float64(f) => f,
            other => panic!("unexpected list elem {other:?}"),
        })
        .collect();
    assert_eq!(nums, vec![1.5, 2.5, 3.5]);

    // --- Boundary-sensitive smart pointers ---
    let boundary_schema = BoundaryPositions::schema().unwrap();
    assert_eq!(
        schema_dtype(&boundary_schema, "opt_box_vec"),
        DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(
        schema_dtype(&boundary_schema, "opt_box_opt"),
        DataType::Int32
    );
    assert_eq!(
        schema_dtype(&boundary_schema, "vec_box_opt"),
        DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(
        schema_dtype(&boundary_schema, "vec_box_vec"),
        DataType::List(Box::new(DataType::List(Box::new(DataType::Int32))))
    );
    assert_eq!(
        schema_dtype(&boundary_schema, "vec_opt_arc_vec"),
        DataType::List(Box::new(DataType::List(Box::new(DataType::Int32))))
    );
    assert_eq!(
        schema_dtype(&boundary_schema, "opt_box_blob"),
        DataType::Binary
    );

    let boundary_rows = vec![
        BoundaryPositions {
            opt_box_vec: Some(Box::new(vec![1, 2])),
            opt_box_opt: Some(Box::new(Some(7))),
            vec_box_opt: vec![Box::new(Some(10)), Box::new(None), Box::new(Some(30))],
            vec_box_vec: vec![Box::new(vec![1, 2]), Box::new(vec![]), Box::new(vec![3])],
            vec_opt_arc_vec: vec![Some(Arc::new(vec![5, 6])), None, Some(Arc::new(vec![]))],
            opt_box_blob: Some(Box::new(b"abc".to_vec())),
        },
        BoundaryPositions {
            opt_box_vec: None,
            opt_box_opt: Some(Box::new(None)),
            vec_box_opt: vec![],
            vec_box_vec: vec![],
            vec_opt_arc_vec: vec![],
            opt_box_blob: None,
        },
        BoundaryPositions {
            opt_box_vec: Some(Box::new(vec![])),
            opt_box_opt: None,
            vec_box_opt: vec![Box::new(None)],
            vec_box_vec: vec![Box::new(vec![9])],
            vec_opt_arc_vec: vec![None],
            opt_box_blob: Some(Box::new(vec![])),
        },
    ];
    let boundary_df = boundary_rows.as_slice().to_dataframe().unwrap();
    assert_eq!(boundary_df.shape(), (3, 6));
    assert_list_i32(&boundary_df, "opt_box_vec", 0, &[Some(1), Some(2)]);
    assert!(matches!(
        boundary_df.column("opt_box_vec").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert_list_i32(&boundary_df, "opt_box_vec", 2, &[]);
    assert!(matches!(
        boundary_df.column("opt_box_opt").unwrap().get(0).unwrap(),
        AnyValue::Int32(7)
    ));
    assert!(matches!(
        boundary_df.column("opt_box_opt").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert!(matches!(
        boundary_df.column("opt_box_opt").unwrap().get(2).unwrap(),
        AnyValue::Null
    ));
    assert_list_i32(&boundary_df, "vec_box_opt", 0, &[Some(10), None, Some(30)]);
    assert_list_i32(&boundary_df, "vec_box_opt", 1, &[]);
    assert_list_i32(&boundary_df, "vec_box_opt", 2, &[None]);
    assert_nested_list_i32(
        &boundary_df,
        "vec_box_vec",
        0,
        &[
            Some(vec![Some(1), Some(2)]),
            Some(vec![]),
            Some(vec![Some(3)]),
        ],
    );
    assert_nested_list_i32(&boundary_df, "vec_box_vec", 1, &[]);
    assert_nested_list_i32(&boundary_df, "vec_box_vec", 2, &[Some(vec![Some(9)])]);
    assert_nested_list_i32(
        &boundary_df,
        "vec_opt_arc_vec",
        0,
        &[Some(vec![Some(5), Some(6)]), None, Some(vec![])],
    );
    assert_nested_list_i32(&boundary_df, "vec_opt_arc_vec", 1, &[]);
    assert_nested_list_i32(&boundary_df, "vec_opt_arc_vec", 2, &[None]);
    match boundary_df.column("opt_box_blob").unwrap().get(0).unwrap() {
        AnyValue::Binary(bytes) => assert_eq!(bytes, b"abc"),
        AnyValue::BinaryOwned(ref bytes) => assert_eq!(bytes.as_slice(), b"abc"),
        other => panic!("unexpected opt_box_blob[0] {other:?}"),
    }
    assert!(matches!(
        boundary_df.column("opt_box_blob").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    match boundary_df.column("opt_box_blob").unwrap().get(2).unwrap() {
        AnyValue::Binary(bytes) => assert_eq!(bytes, b""),
        AnyValue::BinaryOwned(ref bytes) => assert_eq!(bytes.as_slice(), b""),
        other => panic!("unexpected opt_box_blob[2] {other:?}"),
    }

    // --- Regression: as_binary over Box<Vec<u8>> ---
    let regression_schema = Regression::schema().unwrap();
    assert_eq!(
        schema_dtype(&regression_schema, "bx_blob"),
        DataType::Binary
    );
    assert_eq!(schema_dtype(&regression_schema, "arc_date"), DataType::Date);
    assert_eq!(
        schema_dtype(&regression_schema, "bx_chrono_dur"),
        DataType::Duration(TimeUnit::Nanoseconds)
    );

    let reg0 = Regression {
        bx_blob: Box::new(b"binary-blob".to_vec()),
        arc_date: Arc::new(epoch),
        bx_chrono_dur: Box::new(chrono::Duration::nanoseconds(123)),
    };
    let reg_df = reg0.to_dataframe().unwrap();
    assert_eq!(reg_df.shape(), (1, 3));
    match reg_df.column("bx_blob").unwrap().get(0).unwrap() {
        AnyValue::Binary(b) => assert_eq!(b, b"binary-blob"),
        AnyValue::BinaryOwned(ref b) => assert_eq!(b.as_slice(), b"binary-blob"),
        other => panic!("unexpected bx_blob {other:?}"),
    }
    match reg_df.column("arc_date").unwrap().get(0).unwrap() {
        AnyValue::Date(d) => assert_eq!(d, 0),
        other => panic!("unexpected arc_date {other:?}"),
    }

    // --- Empty-DataFrame schema preserved ---
    let empty = Bare::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 7));
    let composed_empty = Composed::empty_dataframe().unwrap();
    assert_eq!(composed_empty.shape(), (0, 4));

    println!("Smart-pointer transparency test passed.");
}
