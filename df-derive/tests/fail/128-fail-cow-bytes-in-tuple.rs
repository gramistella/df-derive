use std::borrow::Cow;

use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    pair: (Cow<'static, [u8]>, i32),
}

fn main() {}
