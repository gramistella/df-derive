use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
#[df_derive(columnar = "core::dataframe::Columnar")]
struct ColumnarOnly {
    id: u32,
}

fn main() {}
