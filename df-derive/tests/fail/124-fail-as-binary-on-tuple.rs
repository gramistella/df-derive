use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct WithAsBinaryOnTuple {
    #[df_derive(as_binary)]
    pair: (Vec<u8>, String),
}

fn main() {}
