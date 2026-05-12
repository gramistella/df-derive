use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
union Bits {
    a: u32,
    b: f32,
}

fn main() {}
