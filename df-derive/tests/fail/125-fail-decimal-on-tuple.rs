use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct WithDecimalOnTuple {
    #[df_derive(decimal(precision = 10, scale = 2))]
    pair: (i32, String),
}

fn main() {}
