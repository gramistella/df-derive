use df_derive::ToDataFrame;
use std::borrow::Cow;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    bs: Cow<'static, [u8]>,
}

fn main() {}
