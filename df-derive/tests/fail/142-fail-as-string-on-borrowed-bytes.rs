use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad<'a> {
    #[df_derive(as_string)]
    bytes: &'a [u8],
}

fn main() {}
