use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(decimal(precision = 38, scale = 18))]
    not_a_decimal: f64,
}

fn main() {}
