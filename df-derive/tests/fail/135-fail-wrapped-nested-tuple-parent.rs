use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    nested: Option<((i32, String), bool)>,
}

fn main() {}
