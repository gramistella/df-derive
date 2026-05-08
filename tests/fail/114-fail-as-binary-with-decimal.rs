use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary, decimal(precision = 18, scale = 4))]
    bytes: Vec<u8>,
}

fn main() {}
