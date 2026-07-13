use df_derive::ToDataFrame;

use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(decimal(precision = 39, scale = 10))]
    amount: Decimal,
}

fn main() {}
