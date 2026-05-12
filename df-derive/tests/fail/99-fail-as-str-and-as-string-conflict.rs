use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Conflict {
    #[df_derive(as_str)]
    #[df_derive(as_string)]
    name: String,
}

fn main() {}
