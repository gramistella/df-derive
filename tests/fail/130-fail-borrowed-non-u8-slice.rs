use df_derive::ToDataFrame;

#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad<'a> {
    items: &'a [i32],
}

fn main() {}
