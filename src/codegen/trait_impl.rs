// SchemaProvider and RowWiseGenerator are referenced via fully-qualified paths
use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote; // for trait methods on Strategy

pub fn generate_trait_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let to_df_trait = &config.to_dataframe_trait_path;
    let (impl_generics, ty_generics, where_clause) =
        super::impl_parts_with_bounds(&ir.generics, config);

    if ir.fields.is_empty() {
        return quote! {
            impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
                fn to_dataframe(&self) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                    use polars::prelude::{NamedFrom, DataFrame, Series};
                    let dummy_series = Series::new("_dummy".into(), &[0i32]);
                    let mut df_with_row = DataFrame::new_infer_height(vec![dummy_series.into()])?;
                    df_with_row.drop_in_place("_dummy")?;
                    Ok(df_with_row)
                }

                fn empty_dataframe() -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                    polars::prelude::DataFrame::new_infer_height(vec![])
                }

                fn schema() -> polars::prelude::PolarsResult<::std::vec::Vec<(::std::string::String, polars::prelude::DataType)>> {
                    Ok(::std::vec::Vec::new())
                }
            }
        };
    }

    let strategies = super::strategy::build_strategies(ir, config);
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
        impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
            fn to_dataframe(&self) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                // Pull traits from polars' prelude into scope (incl. `NamedFrom`,
                // `IntoSeries`, `NewChunkedArray`, `ListBuilderTrait`) so the
                // bulk-emit and list-builder helpers spliced in by
                // `series_creations` can resolve `T::method(...)` paths
                // without having to UFCS-qualify every trait method.
                use polars::prelude::*;
                let mut all_series: Vec<polars::prelude::Column> = Vec::new();
                #(
                    all_series.extend(#series_creations);
                )*
                polars::prelude::DataFrame::new_infer_height(all_series)
            }

            fn empty_dataframe() -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                let mut all_series: Vec<polars::prelude::Column> = Vec::new();
                #(
                    all_series.extend(#empty_series_creations);
                )*
                polars::prelude::DataFrame::new_infer_height(all_series)
            }

            fn schema() -> polars::prelude::PolarsResult<::std::vec::Vec<(::std::string::String, polars::prelude::DataType)>> {
                let mut fields: ::std::vec::Vec<(::std::string::String, polars::prelude::DataType)> = ::std::vec::Vec::new();
                #(
                    fields.extend(#schema_entries);
                )*
                Ok(fields)
            }
        }
    }
}
