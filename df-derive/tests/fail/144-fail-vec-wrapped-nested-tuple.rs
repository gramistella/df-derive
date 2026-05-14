use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    nested: Vec<((i32, String), bool)>,
}

fn main() {}
