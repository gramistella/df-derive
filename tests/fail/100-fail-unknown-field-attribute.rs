use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Typo {
    #[df_derive(as_strg)]
    s: String,
}

fn main() {}
