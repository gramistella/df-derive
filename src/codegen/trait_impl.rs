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
                    Self::__df_derive_columnar_from_refs(&[self])
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
    let empty_series_creations: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::Strategy::gen_empty_series_creation)
        .collect();
    let schema_entries: Vec<TokenStream> = strategies
        .iter()
        .map(super::strategy::Strategy::gen_schema_entries)
        .collect();

    // `to_dataframe(&self)` delegates to the inherent columnar helper with a
    // single-element ref slice. This is the one source of truth for the
    // row-shape logic — there is no parallel per-row codegen path. The
    // bulk-emit branches for nested-leaf, `Option<Inner>`, and `Vec<Inner>`
    // shapes inside `__df_derive_columnar_from_refs` keep the N=1 cost
    // within ~10% of the previous specialized per-row path while removing a
    // whole second family of strategy methods (`gen_series_creation`,
    // `gen_anyvalue_conversion`, the `__df_derive_to_anyvalues` and
    // `__df_derive_collect_vec_as_prefixed_list_series` inherent helpers).
    //
    // `empty_dataframe` and `schema` keep their own codegen because they
    // never take a `&self` — they're shape-only operations. Routing
    // `empty_dataframe` through `__df_derive_columnar_from_refs(&[])` would
    // recurse, since that helper delegates to `empty_dataframe` on empty
    // input.
    quote! {
        impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
            fn to_dataframe(&self) -> #pp::PolarsResult<#pp::DataFrame> {
                Self::__df_derive_columnar_from_refs(&[self])
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
