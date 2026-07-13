use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_str)]
    not_a_string: f64,
}

fn main() {}
