use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
#[df_derive(trait = "core::dataframe::ToDataFrame")]
#[df_derive(trait = "core::dataframe::ToDataFrame")]
struct DuplicateTrait {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(
    trait = "core::dataframe::ToDataFrame",
    columnar = "core::dataframe::Columnar",
)]
#[df_derive(columnar = "core::dataframe::Columnar")]
struct DuplicateColumnar {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(decimal128_encode = "core::dataframe::Decimal128Encode")]
#[df_derive(decimal128_encode = "core::dataframe::Decimal128Encode")]
struct DuplicateDecimal128Encode {
    id: u32,
}

fn main() {}
