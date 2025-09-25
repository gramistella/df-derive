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
