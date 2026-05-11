use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct Blobs {
    #[df_derive(as_binary)]
    blob: Vec<u8>,
    #[df_derive(as_binary)]
    opt_blob: Option<Vec<u8>>,
    #[df_derive(as_binary)]
    blobs: Vec<Vec<u8>>,
    #[df_derive(as_binary)]
    opt_blobs_inner: Vec<Option<Vec<u8>>>,
    #[df_derive(as_binary)]
    opt_blobs_outer: Option<Vec<Vec<u8>>>,
}

// Regression pin: a separate struct whose `Vec<u8>` field has NO attribute.
// The default schema and runtime dtype must remain `List(UInt8)`.
#[derive(ToDataFrame, Clone)]
struct DefaultRaw {
    raw: Vec<u8>,
}

fn assert_col_bytes(df: &DataFrame, col: &str, row: usize, expected: &[u8]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    match v {
        AnyValue::Binary(b) => assert_eq!(b, expected, "col {col} row {row}"),
        AnyValue::BinaryOwned(ref b) => assert_eq!(b.as_slice(), expected, "col {col} row {row}"),
        other => panic!("unexpected AnyValue for {col} row {row}: {other:?}"),
    }
}

fn assert_col_null(df: &DataFrame, col: &str, row: usize) {
    let v = df.column(col).unwrap().get(row).unwrap();
    assert!(
        matches!(v, AnyValue::Null),
        "expected null at {col}[{row}], got {v:?}"
    );
}

fn assert_list_of_bytes(df: &DataFrame, col: &str, row: usize, expected: &[Option<&[u8]>]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    if let AnyValue::List(inner) = v {
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
            expected.iter().map(|o| o.map(|b| b.to_vec())).collect();
        assert_eq!(actual, expected_owned, "col {col} row {row}");
    } else {
        panic!("expected List for {col} row {row}, got {v:?}");
    }
}

fn assert_list_u8(df: &DataFrame, col: &str, row: usize, expected: &[u8]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    if let AnyValue::List(inner) = v {
        let actual: Vec<u8> = inner
            .iter()
            .map(|av| match av {
                AnyValue::UInt8(v) => v,
                other => panic!("unexpected AnyValue inside list {col}: {other:?}"),
            })
            .collect();
        assert_eq!(actual, expected, "col {col} row {row}");
    } else {
        panic!("expected List for {col} row {row}, got {v:?}");
    }
}

fn main() {
    println!("--- Testing #[df_derive(as_binary)] attribute for Vec<u8> shapes ---");

    let small: Vec<u8> = b"hi".to_vec(); // 2 bytes (BinaryView inline)
    let medium: Vec<u8> = b"twelve_bytes".to_vec(); // 12 bytes (boundary inline)
    let large: Vec<u8> = (0..40u8).collect::<Vec<u8>>(); // 40 bytes (out-of-line)

    let row0 = Blobs {
        blob: small.clone(),
        opt_blob: Some(medium.clone()),
        blobs: vec![small.clone(), large.clone()],
        opt_blobs_inner: vec![Some(small.clone()), None, Some(large.clone())],
        opt_blobs_outer: Some(vec![small.clone(), medium.clone()]),
    };
    let row1 = Blobs {
        blob: large.clone(),
        opt_blob: None,
        blobs: vec![],
        opt_blobs_inner: vec![None],
        opt_blobs_outer: None,
    };
    let row2 = Blobs {
        blob: vec![],
        opt_blob: Some(vec![]),
        blobs: vec![vec![]],
        opt_blobs_inner: vec![Some(vec![])],
        opt_blobs_outer: Some(vec![]),
    };

    println!("Single-row to_dataframe path...");
    let df = row0.clone().to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 5));

    let schema = df.schema();
    assert_eq!(schema.get("blob"), Some(&DataType::Binary));
    assert_eq!(schema.get("opt_blob"), Some(&DataType::Binary));
    assert_eq!(
        schema.get("blobs"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );
    assert_eq!(
        schema.get("opt_blobs_inner"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );
    assert_eq!(
        schema.get("opt_blobs_outer"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );

    println!("Default-behavior pin: Vec<u8> with no attribute -> List(UInt8)...");
    let raw_schema = DefaultRaw::schema().unwrap();
    let raw_dtype = raw_schema
        .iter()
        .find_map(|(n, d)| (n == "raw").then(|| d.clone()))
        .unwrap();
    assert_eq!(raw_dtype, DataType::List(Box::new(DataType::UInt8)));
    let raw_df = DefaultRaw {
        raw: vec![0, 1, 255],
    }
    .to_dataframe()
    .unwrap();
    assert_eq!(
        raw_df.schema().get("raw"),
        Some(&DataType::List(Box::new(DataType::UInt8)))
    );
    assert_list_u8(&raw_df, "raw", 0, &[0, 1, 255]);

    assert_col_bytes(&df, "blob", 0, &small);
    assert_col_bytes(&df, "opt_blob", 0, &medium);
    assert_list_of_bytes(
        &df,
        "blobs",
        0,
        &[Some(small.as_slice()), Some(large.as_slice())],
    );
    assert_list_of_bytes(
        &df,
        "opt_blobs_inner",
        0,
        &[Some(small.as_slice()), None, Some(large.as_slice())],
    );
    assert_list_of_bytes(
        &df,
        "opt_blobs_outer",
        0,
        &[Some(small.as_slice()), Some(medium.as_slice())],
    );

    println!("Columnar batch path with mixed Some/None and empty payloads...");
    let batch = vec![row0.clone(), row1.clone(), row2.clone()];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (3, 5));

    // Same dtypes after the columnar path.
    let bschema = df_batch.schema();
    assert_eq!(bschema.get("blob"), Some(&DataType::Binary));
    assert_eq!(bschema.get("opt_blob"), Some(&DataType::Binary));
    assert_eq!(
        bschema.get("blobs"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );
    assert_eq!(
        bschema.get("opt_blobs_inner"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );
    assert_eq!(
        bschema.get("opt_blobs_outer"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );

    // Bare Binary leaf — three rows worth.
    assert_col_bytes(&df_batch, "blob", 0, &small);
    assert_col_bytes(&df_batch, "blob", 1, &large);
    assert_col_bytes(&df_batch, "blob", 2, &[]);

    // Option<Vec<u8>> — Some(>0) / None / Some(empty).
    assert_col_bytes(&df_batch, "opt_blob", 0, &medium);
    assert_col_null(&df_batch, "opt_blob", 1);
    assert_col_bytes(&df_batch, "opt_blob", 2, &[]);

    // Vec<Vec<u8>> — full / empty / single-empty.
    assert_list_of_bytes(
        &df_batch,
        "blobs",
        0,
        &[Some(small.as_slice()), Some(large.as_slice())],
    );
    assert_list_of_bytes(&df_batch, "blobs", 1, &[]);
    assert_list_of_bytes(&df_batch, "blobs", 2, &[Some(&[][..])]);

    // Vec<Option<Vec<u8>>> — interleaved / pure-None / single-Some-empty.
    assert_list_of_bytes(
        &df_batch,
        "opt_blobs_inner",
        0,
        &[Some(small.as_slice()), None, Some(large.as_slice())],
    );
    assert_list_of_bytes(&df_batch, "opt_blobs_inner", 1, &[None]);
    assert_list_of_bytes(&df_batch, "opt_blobs_inner", 2, &[Some(&[][..])]);

    // Option<Vec<Vec<u8>>> — Some / None / Some(empty outer).
    assert_list_of_bytes(
        &df_batch,
        "opt_blobs_outer",
        0,
        &[Some(small.as_slice()), Some(medium.as_slice())],
    );
    assert_col_null(&df_batch, "opt_blobs_outer", 1);
    assert_list_of_bytes(&df_batch, "opt_blobs_outer", 2, &[]);

    println!("Empty-DataFrame schema check...");
    let empty = Blobs::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 5));
    let eschema = empty.schema();
    assert_eq!(eschema.get("blob"), Some(&DataType::Binary));
    assert_eq!(eschema.get("opt_blob"), Some(&DataType::Binary));
    assert_eq!(
        eschema.get("blobs"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );
    assert_eq!(
        eschema.get("opt_blobs_inner"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );
    assert_eq!(
        eschema.get("opt_blobs_outer"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );

    println!("Nested-struct round-trip: outer with an inner struct that opts in...");
    test_nested_round_trip();

    println!("\nas_binary attribute test completed successfully");
}

#[derive(ToDataFrame, Clone)]
struct Inner {
    #[df_derive(as_binary)]
    payload: Vec<u8>,
}

#[derive(ToDataFrame, Clone)]
struct OuterScalar {
    label: String,
    inner: Inner,
}

#[derive(ToDataFrame, Clone)]
struct OuterVec {
    label: String,
    inners: Vec<Inner>,
}

fn test_nested_round_trip() {
    // Scalar nested: outer column for inner.payload should be Binary.
    let outer_scalar = OuterScalar {
        label: "row".into(),
        inner: Inner {
            payload: b"abc".to_vec(),
        },
    };
    let df = outer_scalar.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 2));
    assert_eq!(df.schema().get("inner.payload"), Some(&DataType::Binary));
    assert_col_bytes(&df, "inner.payload", 0, b"abc");

    let batch_scalar = vec![
        OuterScalar {
            label: "a".into(),
            inner: Inner {
                payload: b"abc".to_vec(),
            },
        },
        OuterScalar {
            label: "b".into(),
            inner: Inner {
                payload: vec![],
            },
        },
        OuterScalar {
            label: "c".into(),
            inner: Inner {
                payload: (0..50u8).collect(),
            },
        },
    ];
    let df_batch = batch_scalar.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (3, 2));
    assert_eq!(
        df_batch.schema().get("inner.payload"),
        Some(&DataType::Binary)
    );
    assert_col_bytes(&df_batch, "inner.payload", 0, b"abc");
    assert_col_bytes(&df_batch, "inner.payload", 1, &[]);
    assert_col_bytes(&df_batch, "inner.payload", 2, &(0..50u8).collect::<Vec<_>>());

    // Vec-of-inner nested: outer column for inners.payload should be List(Binary).
    let outer_vec = OuterVec {
        label: "row".into(),
        inners: vec![
            Inner {
                payload: b"x".to_vec(),
            },
            Inner {
                payload: b"yy".to_vec(),
            },
        ],
    };
    let df_vec = outer_vec.to_dataframe().unwrap();
    assert_eq!(
        df_vec.schema().get("inners.payload"),
        Some(&DataType::List(Box::new(DataType::Binary)))
    );
    assert_list_of_bytes(
        &df_vec,
        "inners.payload",
        0,
        &[Some(&b"x"[..]), Some(&b"yy"[..])],
    );
}
