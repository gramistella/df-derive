use df_derive::ToDataFrame;

use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_string, decimal(precision = 18, scale = 6))]
    amount: Decimal,
}

fn main() {}
