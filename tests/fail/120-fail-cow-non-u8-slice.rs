use std::borrow::Cow;

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    items: Cow<'static, [i32]>,
}

fn main() {}
