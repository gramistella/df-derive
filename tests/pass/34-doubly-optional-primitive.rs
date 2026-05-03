// Primitive `Option<Option<T>>` coverage.
//
// Polars cannot represent two distinct null states, so `Some(None)` and the
// outer `None` collapse to the same `AnyValue::Null`. The encoder collapses
// any consecutive run of `Option`s above a primitive leaf into a single
// validity bit; this test pins that behavior for the three representative
// primitive shapes (Copy numeric, owning `String`, transformed `Decimal`).
//
// The schema entry stays `Int32` / `String` / `Decimal(p, s)` (NOT `List`)
// because `Option` doesn't add a list layer — it's nullability folded into
// the column's validity bitmap.

use df_derive::ToDataFrame;
use polars::prelude::*;
use rust_decimal::Decimal;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame)]
struct DoublyOptional {
    nested_int: Option<Option<i32>>,
    nested_string: Option<Option<String>>,
    #[df_derive(decimal(precision = 10, scale = 2))]
    nested_decimal: Option<Option<Decimal>>,
}

fn main() {
    let expected_schema = vec![
        ("nested_int".to_string(), DataType::Int32),
        ("nested_string".to_string(), DataType::String),
        (
            "nested_decimal".to_string(),
            DataType::Decimal(10, 2),
        ),
    ];
    assert_eq!(DoublyOptional::schema().unwrap(), expected_schema);

    let items = vec![
        DoublyOptional {
            nested_int: Some(Some(42)),
            nested_string: Some(Some("alpha".to_string())),
            nested_decimal: Some(Some(Decimal::new(12345, 2))),
        },
        DoublyOptional {
            nested_int: Some(None),
            nested_string: Some(None),
            nested_decimal: Some(None),
        },
        DoublyOptional {
            nested_int: None,
            nested_string: None,
            nested_decimal: None,
        },
    ];

    let df = items.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (3, 3));
    assert_eq!(
        df.get_column_names(),
        vec!["nested_int", "nested_string", "nested_decimal"]
    );

    // Schema dtypes are scalar (no `List` wrap).
    assert_eq!(df.column("nested_int").unwrap().dtype(), &DataType::Int32);
    assert_eq!(
        df.column("nested_string").unwrap().dtype(),
        &DataType::String
    );
    assert_eq!(
        df.column("nested_decimal").unwrap().dtype(),
        &DataType::Decimal(10, 2)
    );

    // Row 0: every Some(Some(v)) round-trips.
    assert_eq!(
        df.column("nested_int").unwrap().get(0).unwrap(),
        AnyValue::Int32(42)
    );
    let s0 = df.column("nested_string").unwrap().get(0).unwrap();
    if let AnyValue::String(s) = s0 {
        assert_eq!(s, "alpha");
    } else {
        panic!("expected string at row 0, got {s0:?}");
    }
    let d0 = df.column("nested_decimal").unwrap().get(0).unwrap();
    if let AnyValue::Decimal(mantissa, _, _) = d0 {
        assert_eq!(mantissa, 12345);
    } else {
        panic!("expected Decimal at row 0, got {d0:?}");
    }

    // Rows 1 and 2: Some(None) and None both surface as AnyValue::Null.
    for row in [1, 2] {
        assert!(matches!(
            df.column("nested_int").unwrap().get(row).unwrap(),
            AnyValue::Null
        ));
        assert!(matches!(
            df.column("nested_string").unwrap().get(row).unwrap(),
            AnyValue::Null
        ));
        assert!(matches!(
            df.column("nested_decimal").unwrap().get(row).unwrap(),
            AnyValue::Null
        ));
    }

    // Empty slice yields an empty DataFrame with the declared schema.
    let empty = <[DoublyOptional] as ToDataFrameVec>::to_dataframe(&[]).unwrap();
    assert_eq!(empty.shape(), (0, 3));

    // Single-row API (`to_dataframe(&self)`) hits the same encoder path.
    let single = DoublyOptional {
        nested_int: Some(Some(99)),
        nested_string: Some(Some("solo".to_string())),
        nested_decimal: Some(Some(Decimal::new(7, 0))),
    };
    let single_df = single.to_dataframe().unwrap();
    assert_eq!(single_df.shape(), (1, 3));
}
