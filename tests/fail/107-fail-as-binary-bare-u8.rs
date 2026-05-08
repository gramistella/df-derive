use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary)]
    one_byte: u8,
}

fn main() {}
