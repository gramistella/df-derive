use std::borrow::Cow;

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    bs: Cow<'static, [u8]>,
}

fn main() {}
