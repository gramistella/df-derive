use df_derive::ToDataFrame;

mod custom_runtime {
    pub trait MyToDataFrame {}
    pub trait MyColumnarTrait {}
}

#[derive(ToDataFrame)]
#[df_derive(
    trait = "df_derive::dataframe::ToDataFrame",
    columnar = "core::dataframe::Columnar"
)]
struct MixedBuiltinCustom {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(
    trait = "df_derive::dataframe::ToDataFrame",
    columnar = "custom_runtime::MyColumnarTrait"
)]
struct BuiltinTraitCustomColumnarName {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(
    trait = "custom_runtime::MyToDataFrame",
    columnar = "df_derive::dataframe::Columnar"
)]
struct CustomTraitNameBuiltinColumnar {
    id: u32,
}

fn main() {}
