use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct WithDecimalOnTuple {
    #[df_derive(decimal(precision = 10, scale = 2))]
    pair: (i32, String),
}

fn main() {}
