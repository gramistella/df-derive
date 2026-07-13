use std::borrow::Cow;

use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    items: Cow<'static, [i32]>,
}

fn main() {}
