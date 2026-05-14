use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct DuplicatePrecision {
    #[df_derive(decimal(precision = 10, precision = 12, scale = 2))]
    amount: Decimal,
}

#[derive(ToDataFrame)]
struct DuplicateScale {
    #[df_derive(decimal(precision = 10, scale = 2, scale = 2))]
    amount: Decimal,
}

fn main() {}
