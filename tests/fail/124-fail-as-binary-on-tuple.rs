use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct WithAsBinaryOnTuple {
    #[df_derive(as_binary)]
    pair: (Vec<u8>, String),
}

fn main() {}
