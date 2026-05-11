use df_derive::ToDataFrame;

#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad<'a> {
    #[df_derive(as_binary)]
    items: &'a [i32],
}

fn main() {}
