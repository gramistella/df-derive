use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary)]
    bytes: Vec<Option<u8>>,
}

fn main() {}
