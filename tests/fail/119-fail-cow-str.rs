use df_derive::ToDataFrame;
use std::borrow::Cow;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    s: Cow<'static, str>,
}

fn main() {}
