mod columnar_impl;
mod common;
mod helpers;
mod strategy;
mod trait_impl;
mod type_registry;
mod wrapped_codegen;
mod wrapper_processor;

use crate::ir::StructIR;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Macro-wide configuration for generated code
pub struct MacroConfig {
    /// Fully-qualified path to the `ToDataFrame` trait (e.g., `paft::dataframe::ToDataFrame`)
    pub to_dataframe_trait_path: TokenStream,
    /// Fully-qualified path to the Columnar trait (e.g., `paft::dataframe::Columnar`)
    pub columnar_trait_path: TokenStream,
}

pub fn resolve_paft_crate_path() -> TokenStream {
    match crate_name("paft") {
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote! { ::#ident::dataframe }
        }
        Ok(FoundCrate::Itself) => {
            quote! { crate::dataframe }
        }
        _ => match crate_name("paft-utils") {
            Ok(FoundCrate::Name(name)) => {
                let ident = format_ident!("{}", name);
                quote! { ::#ident::dataframe }
            }
            Ok(FoundCrate::Itself) => {
                quote! { crate::dataframe }
            }
            _ => {
                quote! { crate::core::dataframe }
            }
        },
    }
}

pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let trait_impl = trait_impl::generate_trait_impl(ir, config);
    let columnar_impl = columnar_impl::generate_columnar_impl(ir, config);
    let helpers_impl = helpers::generate_helpers_impl(ir, config);

    quote! {
        #trait_impl
        #columnar_impl
        #helpers_impl
    }
}

/// Build `impl_generics`, `ty_generics`, and `where_clause` token streams suitable
/// for splicing into an `impl` header. When the struct has type parameters, each
/// one is augmented with the configured `ToDataFrame` and `Columnar` trait
/// bounds so the generated method bodies can call those traits on the params.
pub fn impl_parts_with_bounds(
    generics: &syn::Generics,
    config: &MacroConfig,
) -> (TokenStream, TokenStream, TokenStream) {
    let mut generics = generics.clone();
    let to_df_trait = &config.to_dataframe_trait_path;
    let columnar_trait = &config.columnar_trait_path;
    let to_df_bound: syn::TypeParamBound =
        syn::parse2(quote! { #to_df_trait }).expect("trait path should parse as bound");
    let columnar_bound: syn::TypeParamBound =
        syn::parse2(quote! { #columnar_trait }).expect("trait path should parse as bound");
    // Clone is required because the bulk emitters for generic-typed fields
    // collect a contiguous `Vec<T>` from `&[Self]` before calling
    // `T::columnar_to_dataframe`. Injecting it here turns a cryptic
    // "T: Clone is not satisfied" error inside macro-expanded source into a
    // bound-mismatch error at the user's struct definition.
    let clone_bound: syn::TypeParamBound = syn::parse2(quote! { ::core::clone::Clone })
        .expect("Clone path should parse as bound");
    for tp in generics.type_params_mut() {
        tp.bounds.push(to_df_bound.clone());
        tp.bounds.push(columnar_bound.clone());
        tp.bounds.push(clone_bound.clone());
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    (
        quote! { #impl_generics },
        quote! { #ty_generics },
        quote! { #where_clause },
    )
}
