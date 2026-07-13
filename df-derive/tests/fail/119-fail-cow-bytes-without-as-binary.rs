use std::borrow::Cow;

use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    bs: Cow<'static, [u8]>,
}

fn main() {}
