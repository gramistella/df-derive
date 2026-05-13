use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct BadSkip {
    #[df_derive(skip, as_str)]
    value: String,
}

fn main() {}
