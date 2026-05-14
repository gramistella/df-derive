use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
#[df_derive(columnar = "core::dataframe::Columnar")]
struct ColumnarOnly {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(
    trait = "df_derive::dataframe::ToDataFrame",
    columnar = "core::dataframe::Columnar"
)]
struct MixedBuiltinCustom {
    id: u32,
}

fn main() {}
