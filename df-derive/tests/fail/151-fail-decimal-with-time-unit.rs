use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct BadDecimal {
    #[df_derive(decimal(precision = 10, scale = 2), time_unit = "ns")]
    amount: rust_decimal::Decimal,
}

fn main() {}
