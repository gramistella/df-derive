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

    if ir.columns.is_empty() {
        return quote! {
            #[automatically_derived]
            impl #impl_generics #to_df_trait for #struct_name #ty_generics #where_clause {
                fn to_dataframe(&self) -> #pp::PolarsResult<#pp::DataFrame> {
                    <Self as #columnar_trait>::columnar_from_refs(&[self])
                }

                fn empty_dataframe() -> #pp::PolarsResult<#pp::DataFrame> {
                    ::std::result::Result::Ok(#pp::DataFrame::empty_with_height(0))
                }

                fn schema() -> #pp::PolarsResult<::std::vec::Vec<(::std::string::String, #pp::DataType)>> {
                    ::std::result::Result::Ok(::std::vec::Vec::new())
                }
            }
        };
    }

    let empty_series_creations: Vec<TokenStream> = ir
        .columns
        .iter()
        .map(|column| super::schema::build_empty_series(column, config))
        .collect();
    let schema_entries: Vec<TokenStream> = ir
        .columns
        .iter()
        .map(|column| super::schema::build_schema_entries(column, config))
        .collect();
    let unique_name_validation = if super::support::needs_unique_name_validation(ir) {
        let validate_unique_column_names = super::encoder::idents::validate_unique_column_names();
        quote! {
            #validate_unique_column_names(
                all_series.iter().map(|column| column.name().as_str()),
                ::core::any::type_name::<Self>(),
            )?;
        }
    } else {
        TokenStream::new()
    };
    let schema_unique_name_validation = if super::support::needs_unique_name_validation(ir) {
        let validate_unique_column_names = super::encoder::idents::validate_unique_column_names();
        quote! {
            #validate_unique_column_names(
                fields.iter().map(|(name, _)| name.as_str()),
                ::core::any::type_name::<Self>(),
            )?;
        }
    } else {
        TokenStream::new()
    };

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
                #unique_name_validation
                #pp::DataFrame::new(0, all_series)
            }

            fn schema() -> #pp::PolarsResult<::std::vec::Vec<(::std::string::String, #pp::DataType)>> {
                let mut fields: ::std::vec::Vec<(::std::string::String, #pp::DataType)> = ::std::vec::Vec::new();
                #(
                    fields.extend(#schema_entries);
                )*
                #schema_unique_name_validation
                ::std::result::Result::Ok(fields)
            }
        }
    }
}
