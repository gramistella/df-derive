use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_trait_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let to_df_trait = &config.to_dataframe_trait_path;
    let columnar_trait = &config.columnar_trait_path;
    let pp = super::polars_paths::prelude();
    let (impl_generics, ty_generics, where_clause) =
        super::impl_parts_with_bounds(&ir.generics, config);

    if ir.fields.is_empty() {
        return quote! {
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

                fn to_inner_values(&self) -> #pp::PolarsResult<::std::vec::Vec<#pp::AnyValue<'static>>> {
                    ::std::result::Result::Ok(::std::vec::Vec::new())
                }
            }
        };
    }

    let strategies = super::strategy::build_strategies(ir, config);
    let empty_series_creations: Vec<TokenStream> = strategies
        .iter()
        .map(|s| s.gen_empty_series_creation())
        .collect();
    let schema_entries: Vec<TokenStream> =
        strategies.iter().map(|s| s.gen_schema_entries()).collect();

    let single_it_ident = format_ident!("__df_derive_self_it");
    let out_values_ident = format_ident!("__df_derive_out");
    let single_pushes: Vec<TokenStream> = strategies
        .iter()
        .map(|s| s.gen_for_anyvalue(&single_it_ident, &out_values_ident))
        .collect();

    // `to_dataframe(&self)` delegates to the `Columnar::columnar_from_refs`
    // trait method with a single-element ref slice. There is no parallel
    // per-row codegen path — the trait method is the one source of truth
    // for row-shape logic (with bulk-emit branches for nested-leaf,
    // `Option<Inner>`, and `Vec<Inner>` shapes that keep the N=1 cost
    // within ~10% of the previous specialized per-row path).
    //
    // `empty_dataframe` and `schema` keep their own codegen because they
    // never take a `&self` — they're shape-only operations. Routing
    // `empty_dataframe` through `columnar_from_refs(&[])` would recurse,
    // since that helper delegates to `empty_dataframe` on empty input.
    //
    // `to_inner_values(&self)` overrides the trait's default (which round-
    // trips through `to_dataframe()`) with a direct per-field push into
    // `Vec<AnyValue>`. Used by parent nested-leaf paths so a chain of
    // concrete or generic types never pays for a one-row `DataFrame`
    // allocation per element.
    quote! {
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

            fn to_inner_values(&self) -> #pp::PolarsResult<::std::vec::Vec<#pp::AnyValue<'static>>> {
                // Per-field codegen below may emit bare `Inner::schema()`
                // calls (e.g. on a nested `Option<Inner>`'s `None` arm).
                // Bring the trait into scope so trait-method dispatch works
                // inside this impl body.
                #[allow(unused_imports)]
                use #to_df_trait as _;
                let #single_it_ident = self;
                let mut #out_values_ident: ::std::vec::Vec<#pp::AnyValue<'static>> = ::std::vec::Vec::new();
                #(#single_pushes)*
                ::std::result::Result::Ok(#out_values_ident)
            }
        }
    }
}
