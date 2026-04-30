use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
enum Status {
    Active,
    Inactive,
}

fn main() {}
