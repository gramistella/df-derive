use df_derive::ToDataFrame;

#[path = "../common.rs"]
mod core;

struct Inner {
    value: i32,
}

#[derive(ToDataFrame)]
struct Outer {
    inner: Inner,
}

struct MyDecimal {
    mantissa: i128,
}

#[derive(ToDataFrame)]
struct Tx {
    #[df_derive(decimal(precision = 20, scale = 4))]
    amount: MyDecimal,
}

fn main() {}
