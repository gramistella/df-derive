use df_derive::ToDataFrame;

#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad<'a> {
    bs: &'a [u8],
}

fn main() {}
