use std::num::{
    NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize, NonZeroU8,
    NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize,
};
use std::sync::Arc;

use df_derive::ToDataFrame;
use polars::prelude::*;

#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct NonZeroScalars {
    i8_v: NonZeroI8,
    i16_v: NonZeroI16,
    i32_v: NonZeroI32,
    i64_v: NonZeroI64,
    i128_v: NonZeroI128,
    isize_v: NonZeroIsize,
    u8_v: NonZeroU8,
    u16_v: NonZeroU16,
    u32_v: NonZeroU32,
    u64_v: NonZeroU64,
    u128_v: NonZeroU128,
    usize_v: NonZeroUsize,
    opt_i32: Option<NonZeroI32>,
    opt_u64: Option<std::num::NonZeroU64>,
    pair: (NonZeroU8, NonZeroIsize),
    opt_pair: Option<(NonZeroI16, NonZeroU16)>,
}

#[derive(ToDataFrame, Clone)]
struct NonZeroVecs {
    ids: Vec<NonZeroU32>,
    nullable_counts: Vec<Option<NonZeroI64>>,
    nested_flags: Vec<Vec<NonZeroU8>>,
    tuple_items: Vec<(NonZeroU16, NonZeroIsize)>,
    arc_values: Vec<Arc<NonZeroU32>>,
    boxed_opt: Option<Box<NonZeroI32>>,
}

fn nz_i8(v: i8) -> NonZeroI8 {
    NonZeroI8::new(v).unwrap()
}

fn nz_i16(v: i16) -> NonZeroI16 {
    NonZeroI16::new(v).unwrap()
}

fn nz_i32(v: i32) -> NonZeroI32 {
    NonZeroI32::new(v).unwrap()
}

fn nz_i64(v: i64) -> NonZeroI64 {
    NonZeroI64::new(v).unwrap()
}

fn nz_i128(v: i128) -> NonZeroI128 {
    NonZeroI128::new(v).unwrap()
}

fn nz_isize(v: isize) -> NonZeroIsize {
    NonZeroIsize::new(v).unwrap()
}

fn nz_u8(v: u8) -> NonZeroU8 {
    NonZeroU8::new(v).unwrap()
}

fn nz_u16(v: u16) -> NonZeroU16 {
    NonZeroU16::new(v).unwrap()
}

fn nz_u32(v: u32) -> NonZeroU32 {
    NonZeroU32::new(v).unwrap()
}

fn nz_u64(v: u64) -> NonZeroU64 {
    NonZeroU64::new(v).unwrap()
}

fn nz_u128(v: u128) -> NonZeroU128 {
    NonZeroU128::new(v).unwrap()
}

fn nz_usize(v: usize) -> NonZeroUsize {
    NonZeroUsize::new(v).unwrap()
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(name, _)| name == col)
        .map(|(_, dtype)| dtype.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}

fn assert_list_u32(df: &DataFrame, col: &str, row: usize, expected: &[u32]) {
    let AnyValue::List(inner) = df.column(col).unwrap().get(row).unwrap() else {
        panic!("expected list for {col}[{row}]");
    };
    let actual: Vec<u32> = inner
        .iter()
        .map(|av| match av {
            AnyValue::UInt32(v) => v,
            other => panic!("unexpected list value for {col}: {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected);
}

fn assert_list_i64(df: &DataFrame, col: &str, row: usize, expected: &[Option<i64>]) {
    let AnyValue::List(inner) = df.column(col).unwrap().get(row).unwrap() else {
        panic!("expected list for {col}[{row}]");
    };
    let actual: Vec<Option<i64>> = inner
        .iter()
        .map(|av| match av {
            AnyValue::Int64(v) => Some(v),
            AnyValue::Null => None,
            other => panic!("unexpected list value for {col}: {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected);
}

fn main() {
    let scalar_schema = NonZeroScalars::schema().unwrap();
    assert_eq!(schema_dtype(&scalar_schema, "i8_v"), DataType::Int8);
    assert_eq!(schema_dtype(&scalar_schema, "i16_v"), DataType::Int16);
    assert_eq!(schema_dtype(&scalar_schema, "i32_v"), DataType::Int32);
    assert_eq!(schema_dtype(&scalar_schema, "i64_v"), DataType::Int64);
    assert_eq!(schema_dtype(&scalar_schema, "i128_v"), DataType::Int128);
    assert_eq!(schema_dtype(&scalar_schema, "isize_v"), DataType::Int64);
    assert_eq!(schema_dtype(&scalar_schema, "u8_v"), DataType::UInt8);
    assert_eq!(schema_dtype(&scalar_schema, "u16_v"), DataType::UInt16);
    assert_eq!(schema_dtype(&scalar_schema, "u32_v"), DataType::UInt32);
    assert_eq!(schema_dtype(&scalar_schema, "u64_v"), DataType::UInt64);
    assert_eq!(schema_dtype(&scalar_schema, "u128_v"), DataType::UInt128);
    assert_eq!(schema_dtype(&scalar_schema, "usize_v"), DataType::UInt64);
    assert_eq!(schema_dtype(&scalar_schema, "opt_i32"), DataType::Int32);
    assert_eq!(schema_dtype(&scalar_schema, "opt_u64"), DataType::UInt64);
    assert_eq!(schema_dtype(&scalar_schema, "pair.field_0"), DataType::UInt8);
    assert_eq!(
        schema_dtype(&scalar_schema, "pair.field_1"),
        DataType::Int64
    );
    assert_eq!(
        schema_dtype(&scalar_schema, "opt_pair.field_0"),
        DataType::Int16
    );
    assert_eq!(
        schema_dtype(&scalar_schema, "opt_pair.field_1"),
        DataType::UInt16
    );

    let rows = vec![
        NonZeroScalars {
            i8_v: nz_i8(-8),
            i16_v: nz_i16(-16),
            i32_v: nz_i32(-32),
            i64_v: nz_i64(-64),
            i128_v: nz_i128(-128),
            isize_v: nz_isize(-7),
            u8_v: nz_u8(8),
            u16_v: nz_u16(16),
            u32_v: nz_u32(32),
            u64_v: nz_u64(64),
            u128_v: nz_u128(128),
            usize_v: nz_usize(9),
            opt_i32: Some(nz_i32(123)),
            opt_u64: Some(nz_u64(456)),
            pair: (nz_u8(3), nz_isize(-4)),
            opt_pair: Some((nz_i16(-5), nz_u16(6))),
        },
        NonZeroScalars {
            i8_v: nz_i8(1),
            i16_v: nz_i16(2),
            i32_v: nz_i32(3),
            i64_v: nz_i64(4),
            i128_v: nz_i128(5),
            isize_v: nz_isize(6),
            u8_v: nz_u8(1),
            u16_v: nz_u16(2),
            u32_v: nz_u32(3),
            u64_v: nz_u64(4),
            u128_v: nz_u128(5),
            usize_v: nz_usize(6),
            opt_i32: None,
            opt_u64: None,
            pair: (nz_u8(7), nz_isize(8)),
            opt_pair: None,
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (2, 18));
    assert_eq!(df.column("i8_v").unwrap().get(0).unwrap(), AnyValue::Int8(-8));
    assert_eq!(
        df.column("i128_v").unwrap().get(0).unwrap(),
        AnyValue::Int128(-128)
    );
    assert_eq!(
        df.column("usize_v").unwrap().get(0).unwrap(),
        AnyValue::UInt64(9)
    );
    assert_eq!(
        df.column("opt_i32").unwrap().get(0).unwrap(),
        AnyValue::Int32(123)
    );
    assert!(matches!(
        df.column("opt_u64").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert_eq!(
        df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::Int64(-4)
    );
    assert!(matches!(
        df.column("opt_pair.field_0").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));

    let vec_schema = NonZeroVecs::schema().unwrap();
    assert_eq!(
        schema_dtype(&vec_schema, "ids"),
        DataType::List(Box::new(DataType::UInt32))
    );
    assert_eq!(
        schema_dtype(&vec_schema, "nullable_counts"),
        DataType::List(Box::new(DataType::Int64))
    );
    assert_eq!(
        schema_dtype(&vec_schema, "nested_flags"),
        DataType::List(Box::new(DataType::List(Box::new(DataType::UInt8))))
    );
    assert_eq!(
        schema_dtype(&vec_schema, "tuple_items.field_0"),
        DataType::List(Box::new(DataType::UInt16))
    );
    assert_eq!(
        schema_dtype(&vec_schema, "tuple_items.field_1"),
        DataType::List(Box::new(DataType::Int64))
    );
    assert_eq!(
        schema_dtype(&vec_schema, "arc_values"),
        DataType::List(Box::new(DataType::UInt32))
    );
    assert_eq!(schema_dtype(&vec_schema, "boxed_opt"), DataType::Int32);

    let vec_rows = vec![
        NonZeroVecs {
            ids: vec![nz_u32(10), nz_u32(20)],
            nullable_counts: vec![Some(nz_i64(-1)), None, Some(nz_i64(3))],
            nested_flags: vec![vec![nz_u8(1), nz_u8(2)], vec![nz_u8(3)]],
            tuple_items: vec![(nz_u16(11), nz_isize(-11)), (nz_u16(12), nz_isize(12))],
            arc_values: vec![Arc::new(nz_u32(77)), Arc::new(nz_u32(88))],
            boxed_opt: Some(Box::new(nz_i32(-99))),
        },
        NonZeroVecs {
            ids: vec![],
            nullable_counts: vec![None],
            nested_flags: vec![],
            tuple_items: vec![],
            arc_values: vec![],
            boxed_opt: None,
        },
    ];
    let vec_df = vec_rows.as_slice().to_dataframe().unwrap();
    assert_eq!(vec_df.shape(), (2, 7));
    assert_list_u32(&vec_df, "ids", 0, &[10, 20]);
    assert_list_i64(&vec_df, "nullable_counts", 0, &[Some(-1), None, Some(3)]);
    assert_list_u32(&vec_df, "arc_values", 0, &[77, 88]);
    assert_eq!(
        vec_df.column("boxed_opt").unwrap().get(0).unwrap(),
        AnyValue::Int32(-99)
    );
    assert!(matches!(
        vec_df.column("boxed_opt").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));

    let empty = <[NonZeroVecs] as ToDataFrameVec>::to_dataframe(&[]).unwrap();
    assert_eq!(empty.shape(), (0, 7));
}
