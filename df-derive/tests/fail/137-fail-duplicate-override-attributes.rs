use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct DuplicateDecimal {
    #[df_derive(decimal(precision = 10, scale = 2))]
    #[df_derive(decimal(precision = 20, scale = 5))]
    amount: Decimal,
}

#[derive(ToDataFrame)]
struct DuplicateTimeUnit {
    #[df_derive(time_unit = "ms")]
    #[df_derive(time_unit = "ns")]
    ts: DateTime<Utc>,
}

#[derive(ToDataFrame)]
struct DuplicateAsStr {
    #[df_derive(as_str)]
    #[df_derive(as_str)]
    name: String,
}

#[derive(ToDataFrame)]
struct DuplicateAsString {
    #[df_derive(as_string)]
    #[df_derive(as_string)]
    name: String,
}

fn main() {}
