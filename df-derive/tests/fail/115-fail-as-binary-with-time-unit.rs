use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary, time_unit = "ns")]
    bytes: Vec<u8>,
}

fn main() {}
