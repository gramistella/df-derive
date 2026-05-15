use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_trait_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let to_df_trait = &config.traits.to_dataframe;
    let columnar_trait = &config.traits.columnar;
    let pp = config.external_paths.prelude();
    let (impl_generics, ty_generics, where_clause) =
        super::bounds::impl_parts_with_bounds(ir, config);

    if ir.fields.is_empty() {
        return quote! {
            #[automatically_derived]
            impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
                fn to_dataframe(&self) -> #pp::PolarsResult<#pp::DataFrame> {
                    <Self as #columnar_trait>::columnar_from_refs(&[self])
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

    let empty_series_creations: Vec<TokenStream> = ir
        .fields
        .iter()
        .map(|field| super::strategy::build_empty_series(field, config))
        .collect();
    let schema_entries: Vec<TokenStream> = ir
        .fields
        .iter()
        .map(|field| super::strategy::build_schema_entries(field, config))
        .collect();

    // `to_dataframe(&self)` delegates to the `Columnar::columnar_from_refs`
    // trait method with a single-element ref slice. There is no parallel
    // per-row codegen path — the trait method is the one source of truth
    // for row-shape logic. Bulk-emit branches for nested-leaf,
    // `Option<Inner>`, and `Vec<Inner>` shapes keep the N=1 cost bounded.
    //
    // `empty_dataframe` and `schema` keep their own codegen because they
    // never take a `&self` — they're shape-only operations. Routing
    // `empty_dataframe` through `columnar_from_refs(&[])` would recurse,
    // since that helper delegates to `empty_dataframe` on empty input.
    quote! {
        #[automatically_derived]
        impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
            fn to_dataframe(&self) -> #pp::PolarsResult<#pp::DataFrame> {
                <Self as #columnar_trait>::columnar_from_refs(&[self])
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
