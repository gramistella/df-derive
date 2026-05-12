use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct WithAsStringOnTuple {
    #[df_derive(as_string)]
    pair: (String, i32),
}

fn main() {}
