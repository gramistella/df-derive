use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    nested: Vec<((i32, String), bool)>,
}

fn main() {}
