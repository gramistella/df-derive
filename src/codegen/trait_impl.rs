use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_trait_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let to_df_trait = &config.to_dataframe_trait_path;
    let pp = super::polars_paths::prelude();
    let (impl_generics, ty_generics, where_clause) =
        super::impl_parts_with_bounds(&ir.generics, config);

    if ir.fields.is_empty() {
        return quote! {
            impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
                fn to_dataframe(&self) -> #pp::PolarsResult<#pp::DataFrame> {
                    let dummy_series = <#pp::Series as #pp::NamedFrom<_, _>>::new(
                        "_dummy".into(),
                        &[0i32],
                    );
                    let mut df_with_row = #pp::DataFrame::new_infer_height(
                        ::std::vec![dummy_series.into()],
                    )?;
                    df_with_row.drop_in_place("_dummy")?;
                    ::std::result::Result::Ok(df_with_row)
                }

                fn empty_dataframe() -> #pp::PolarsResult<#pp::DataFrame> {
                    #pp::DataFrame::new_infer_height(::std::vec![])
                }

                fn schema() -> #pp::PolarsResult<::std::vec::Vec<(::std::string::String, #pp::DataType)>> {
                    ::std::result::Result::Ok(::std::vec::Vec::new())
                }
            }
        };
    }

    let strategies = super::strategy::build_strategies(ir, config);
    let series_creations: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::Strategy::gen_series_creation)
        .collect();
    let empty_series_creations: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::Strategy::gen_empty_series_creation)
        .collect();
    let schema_entries: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::Strategy::gen_schema_entries)
        .collect();

    quote! {
        impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
            fn to_dataframe(&self) -> #pp::PolarsResult<#pp::DataFrame> {
                let mut all_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
                #(
                    all_series.extend(#series_creations);
                )*
                #pp::DataFrame::new_infer_height(all_series)
            }

            fn empty_dataframe() -> #pp::PolarsResult<#pp::DataFrame> {
                let mut all_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
                #(
                    all_series.extend(#empty_series_creations);
                )*
                #pp::DataFrame::new_infer_height(all_series)
            }

            fn schema() -> #pp::PolarsResult<::std::vec::Vec<(::std::string::String, #pp::DataType)>> {
                let mut fields: ::std::vec::Vec<(::std::string::String, #pp::DataType)> = ::std::vec::Vec::new();
                #(
                    fields.extend(#schema_entries);
                )*
                ::std::result::Result::Ok(fields)
            }
        }
    }
}
