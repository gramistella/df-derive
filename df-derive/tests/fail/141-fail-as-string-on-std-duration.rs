use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_string)]
    duration: std::time::Duration,
}

fn main() {}
