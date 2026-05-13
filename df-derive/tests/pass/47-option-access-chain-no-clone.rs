// Regression test for non-trivial primitive Option access chains:
// `Option<Box<T>>` and `Option<Option<T>>` must not synthesize a hidden
// `T: Clone` bound when the leaf can be consumed by reference.

use chrono::{TimeZone, Utc};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::fmt;

#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{Decimal128Encode, ToDataFrameVec};

struct NoCloneDisplay(&'static str);

impl fmt::Display for NoCloneDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

struct NoCloneDecimal {
    mantissa: i128,
    scale: u32,
}

impl NoCloneDecimal {
    const fn new(mantissa: i128, scale: u32) -> Self {
        Self { mantissa, scale }
    }
}

impl Decimal128Encode for NoCloneDecimal {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        match self.scale.cmp(&target_scale) {
            std::cmp::Ordering::Equal => Some(self.mantissa),
            std::cmp::Ordering::Less => self
                .mantissa
                .checked_mul(10_i128.pow(target_scale - self.scale)),
            std::cmp::Ordering::Greater => {
                Some(self.mantissa / 10_i128.pow(self.scale - target_scale))
            }
        }
    }
}

#[derive(ToDataFrame)]
struct Row<T, D> {
    plain_string_box: Option<Box<String>>,
    plain_string_nested: Option<Option<String>>,
    #[df_derive(as_string)]
    display_box: Option<Box<T>>,
    #[df_derive(as_string)]
    display_nested: Option<Option<T>>,
    #[df_derive(decimal(precision = 12, scale = 2))]
    decimal_box: Option<Box<D>>,
    #[df_derive(decimal(precision = 12, scale = 2))]
    decimal_nested: Option<Option<D>>,
    #[df_derive(as_binary)]
    bytes_box: Option<Box<Vec<u8>>>,
    #[df_derive(time_unit = "ms")]
    datetime_box: Option<Box<chrono::DateTime<Utc>>>,
    #[df_derive(time_unit = "ms")]
    datetime_nested: Option<Option<chrono::DateTime<Utc>>>,
}

fn main() {
    let timestamp = Utc.timestamp_millis_opt(1_700_000_000_123).unwrap();
    let rows = vec![
        Row::<NoCloneDisplay, NoCloneDecimal> {
            plain_string_box: Some(Box::new("boxed string".to_string())),
            plain_string_nested: Some(Some("nested string".to_string())),
            display_box: Some(Box::new(NoCloneDisplay("boxed"))),
            display_nested: Some(Some(NoCloneDisplay("nested"))),
            decimal_box: Some(Box::new(NoCloneDecimal::new(123, 0))),
            decimal_nested: Some(Some(NoCloneDecimal::new(456, 1))),
            bytes_box: Some(Box::new(b"abc".to_vec())),
            datetime_box: Some(Box::new(timestamp)),
            datetime_nested: Some(Some(timestamp)),
        },
        Row {
            plain_string_box: None,
            plain_string_nested: Some(None),
            display_box: None,
            display_nested: Some(None),
            decimal_box: None,
            decimal_nested: None,
            bytes_box: None,
            datetime_box: None,
            datetime_nested: None,
        },
    ];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (2, 9));
    assert_eq!(df.column("plain_string_box").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("plain_string_nested").unwrap().dtype(),
        &DataType::String
    );
    assert_eq!(df.column("display_box").unwrap().dtype(), &DataType::String);
    assert_eq!(df.column("display_nested").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("decimal_box").unwrap().dtype(),
        &DataType::Decimal(12, 2)
    );
    assert_eq!(
        df.column("decimal_nested").unwrap().dtype(),
        &DataType::Decimal(12, 2)
    );
    assert_eq!(df.column("bytes_box").unwrap().dtype(), &DataType::Binary);
    assert_eq!(
        df.column("datetime_box").unwrap().dtype(),
        &DataType::Datetime(TimeUnit::Milliseconds, None)
    );
    assert_eq!(
        df.column("datetime_nested").unwrap().dtype(),
        &DataType::Datetime(TimeUnit::Milliseconds, None)
    );
}
