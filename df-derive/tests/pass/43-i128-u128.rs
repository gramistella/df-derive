// `i128` / `u128` coverage - Polars represents these via native
// `Int128` / `UInt128` lanes when the corresponding Polars dtype features
// are enabled. This pins the same primitive shapes as the smaller integer
// family: bare, Option, Vec, Vec<Option<_>>, Vec<Vec<_>>, and tuple fields.

use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct Int128Bare {
    signed: i128,
    unsigned: u128,
    maybe_signed: Option<i128>,
    maybe_unsigned: Option<u128>,
    pair: (i128, u128),
}

#[derive(ToDataFrame, Clone)]
struct Int128Vecs {
    signed_items: Vec<i128>,
    nullable_signed_items: Vec<Option<i128>>,
    nested_unsigned_items: Vec<Vec<u128>>,
    unsigned_items: Vec<u128>,
}

fn assert_dtype(df: &DataFrame, col: &str, dtype: DataType) {
    assert_eq!(df.column(col).unwrap().dtype(), &dtype);
}

fn assert_list_i128(df: &DataFrame, col: &str, row: usize, expected: &[Option<i128>]) {
    let value = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(series) = value else {
        panic!("expected List for {col}[{row}], got {value:?}");
    };
    let actual: Vec<Option<i128>> = series
        .iter()
        .map(|av| match av {
            AnyValue::Int128(v) => Some(v),
            AnyValue::Null => None,
            other => panic!("expected Int128 or Null, got {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected);
}

fn assert_list_u128(df: &DataFrame, col: &str, row: usize, expected: &[Option<u128>]) {
    let value = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(series) = value else {
        panic!("expected List for {col}[{row}], got {value:?}");
    };
    let actual: Vec<Option<u128>> = series
        .iter()
        .map(|av| match av {
            AnyValue::UInt128(v) => Some(v),
            AnyValue::Null => None,
            other => panic!("expected UInt128 or Null, got {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected);
}

fn main() {
    let bare_schema = Int128Bare::schema().unwrap();
    assert_eq!(
        bare_schema,
        vec![
            ("signed".to_string(), DataType::Int128),
            ("unsigned".to_string(), DataType::UInt128),
            ("maybe_signed".to_string(), DataType::Int128),
            ("maybe_unsigned".to_string(), DataType::UInt128),
            ("pair.field_0".to_string(), DataType::Int128),
            ("pair.field_1".to_string(), DataType::UInt128),
        ]
    );

    let big_signed = i128::MIN + 123_456_789;
    let big_unsigned = u128::MAX - 987_654_321;
    let rows = vec![
        Int128Bare {
            signed: big_signed,
            unsigned: big_unsigned,
            maybe_signed: Some(-42),
            maybe_unsigned: Some(42),
            pair: (i128::MAX, u128::MAX),
        },
        Int128Bare {
            signed: 0,
            unsigned: 0,
            maybe_signed: None,
            maybe_unsigned: None,
            pair: (i128::MIN, u128::MIN),
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (2, 6));
    assert_dtype(&df, "signed", DataType::Int128);
    assert_dtype(&df, "unsigned", DataType::UInt128);
    assert_dtype(&df, "maybe_signed", DataType::Int128);
    assert_dtype(&df, "maybe_unsigned", DataType::UInt128);
    assert_dtype(&df, "pair.field_0", DataType::Int128);
    assert_dtype(&df, "pair.field_1", DataType::UInt128);
    assert_eq!(
        df.column("signed").unwrap().get(0).unwrap(),
        AnyValue::Int128(big_signed)
    );
    assert_eq!(
        df.column("unsigned").unwrap().get(0).unwrap(),
        AnyValue::UInt128(big_unsigned)
    );
    assert_eq!(
        df.column("maybe_signed").unwrap().get(0).unwrap(),
        AnyValue::Int128(-42)
    );
    assert!(matches!(
        df.column("maybe_unsigned").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert_eq!(
        df.column("pair.field_0").unwrap().get(0).unwrap(),
        AnyValue::Int128(i128::MAX)
    );
    assert_eq!(
        df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::UInt128(u128::MAX)
    );

    let vec_schema = Int128Vecs::schema().unwrap();
    assert_eq!(
        vec_schema,
        vec![
            (
                "signed_items".to_string(),
                DataType::List(Box::new(DataType::Int128)),
            ),
            (
                "nullable_signed_items".to_string(),
                DataType::List(Box::new(DataType::Int128)),
            ),
            (
                "nested_unsigned_items".to_string(),
                DataType::List(Box::new(DataType::List(Box::new(DataType::UInt128)))),
            ),
            (
                "unsigned_items".to_string(),
                DataType::List(Box::new(DataType::UInt128)),
            ),
        ]
    );

    let vec_rows = vec![
        Int128Vecs {
            signed_items: vec![i128::MIN, -1, i128::MAX],
            nullable_signed_items: vec![Some(7), None, Some(big_signed)],
            nested_unsigned_items: vec![vec![0, 1], vec![u128::MAX]],
            unsigned_items: vec![u128::MAX, 9],
        },
        Int128Vecs {
            signed_items: vec![],
            nullable_signed_items: vec![],
            nested_unsigned_items: vec![],
            unsigned_items: vec![],
        },
    ];
    let vec_df = vec_rows.as_slice().to_dataframe().unwrap();
    assert_eq!(vec_df.shape(), (2, 4));
    assert_dtype(
        &vec_df,
        "signed_items",
        DataType::List(Box::new(DataType::Int128)),
    );
    assert_dtype(
        &vec_df,
        "unsigned_items",
        DataType::List(Box::new(DataType::UInt128)),
    );
    assert_dtype(
        &vec_df,
        "nested_unsigned_items",
        DataType::List(Box::new(DataType::List(Box::new(DataType::UInt128)))),
    );
    assert_list_i128(
        &vec_df,
        "signed_items",
        0,
        &[Some(i128::MIN), Some(-1), Some(i128::MAX)],
    );
    assert_list_i128(
        &vec_df,
        "nullable_signed_items",
        0,
        &[Some(7), None, Some(big_signed)],
    );
    assert_list_u128(
        &vec_df,
        "unsigned_items",
        0,
        &[Some(u128::MAX), Some(9)],
    );

    let empty_bare = <[Int128Bare] as ToDataFrameVec>::to_dataframe(&[]).unwrap();
    assert_eq!(empty_bare.shape(), (0, 6));
    let empty_vecs = <[Int128Vecs] as ToDataFrameVec>::to_dataframe(&[]).unwrap();
    assert_eq!(empty_vecs.shape(), (0, 4));
}
