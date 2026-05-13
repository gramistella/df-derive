use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    nested: Option<((i32, String), bool)>,
}

fn main() {}
