use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_string, decimal(precision = 18, scale = 6))]
    amount: Decimal,
}

fn main() {}
