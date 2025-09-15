// SchemaProvider and RowWiseGenerator are referenced via fully-qualified paths
use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote; // for trait methods on Strategy

pub fn generate_trait_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let to_df_trait = &config.to_dataframe_trait_path;

    if ir.fields.is_empty() {
        return quote! {
            impl #to_df_trait for #struct_name {
                fn to_dataframe(&self) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                    use polars::prelude::{NamedFrom, DataFrame, Series};
                    let dummy_series = Series::new("_dummy".into(), &[0i32]);
                    let mut df_with_row = DataFrame::new(vec![dummy_series.into()])?;
                    df_with_row.drop_in_place("_dummy")?;
                    Ok(df_with_row)
                }

                fn empty_dataframe() -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                    polars::prelude::DataFrame::new(vec![])
                }

                fn schema() -> polars::prelude::PolarsResult<Vec<(&'static str, polars::prelude::DataType)>> {
                    Ok(Vec::new())
                }
            }
        };
    }

    let strategies = super::strategy::build_strategies(ir);
    let series_creations: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::RowWiseGenerator::gen_series_creation)
        .collect();
    let empty_series_creations: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::RowWiseGenerator::gen_empty_series_creation)
        .collect();
    let schema_entries: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::SchemaProvider::gen_schema_entries)
        .collect();

    quote! {
        impl #to_df_trait for #struct_name {
            fn to_dataframe(&self) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                use polars::prelude::NamedFrom;
                let mut all_series: Vec<polars::prelude::Column> = Vec::new();
                #(
                    all_series.extend(#series_creations);
                )*
                polars::prelude::DataFrame::new(all_series)
            }

            fn empty_dataframe() -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                let mut all_series: Vec<polars::prelude::Column> = Vec::new();
                #(
                    all_series.extend(#empty_series_creations);
                )*
                polars::prelude::DataFrame::new(all_series)
            }

            fn schema() -> polars::prelude::PolarsResult<Vec<(&'static str, polars::prelude::DataType)>> {
                let mut fields: Vec<(&'static str, polars::prelude::DataType)> = Vec::new();
                #(
                    fields.extend(#schema_entries);
                )*
                Ok(fields)
            }
        }
    }
}
