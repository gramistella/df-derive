use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct WithUnit {
    nothing: (),
}

fn main() {}
