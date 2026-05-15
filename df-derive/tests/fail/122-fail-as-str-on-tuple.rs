use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct WithAsStrOnTuple {
    #[df_derive(as_str)]
    pair: (String, i32),
}

#[derive(ToDataFrame)]
struct WithAsStrOnStringTuple {
    #[df_derive(as_str)]
    tuple: (String, String),
}

fn main() {}
