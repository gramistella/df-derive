//! Procedural macros for paft.
//!
//! Currently provides `#[derive(ToDataFrame)]` to convert structs into Polars
//! `DataFrames` with flattened columns, and implements an internal columnar
//! path used by an extension trait to convert slices/Vecs.
#![warn(missing_docs)]
extern crate proc_macro;

mod codegen;
mod ir;
mod parser;
mod type_analysis;
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Derive macro that implements `paft::dataframe::ToDataFrame` for a struct and
/// `paft::dataframe::Columnar` for the same struct, enabling fast conversion of
/// slices/Vecs via the extension trait.
#[proc_macro_derive(ToDataFrame, attributes(df_derive))]
pub fn to_dataframe_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let ast = parse_macro_input!(input as DeriveInput);
    // Parse helper attribute configuration (trait paths)
    let default_df_mod = codegen::resolve_paft_crate_path();
    let mut to_df_trait_path_ts = quote! { #default_df_mod::ToDataFrame };
    let mut columnar_trait_path_ts = quote! { #default_df_mod::Columnar };

    for attr in &ast.attrs {
        if attr.path().is_ident("df_derive") {
            let parse_res = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("trait") {
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    let path: syn::Path = syn::parse_str(&lit.value())
                        .map_err(|e| meta.error(format!("invalid trait path: {e}")))?;
                    to_df_trait_path_ts = quote! { #path };

                    // Automatically infer the Columnar trait path by replacing the final segment
                    let mut columnar_path = path;
                    if let Some(last_segment) = columnar_path.segments.last_mut() {
                        last_segment.ident = syn::Ident::new("Columnar", last_segment.ident.span());
                    }
                    columnar_trait_path_ts = quote! { #columnar_path };
                    Ok(())
                } else if meta.path.is_ident("columnar") {
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    let path: syn::Path = syn::parse_str(&lit.value())
                        .map_err(|e| meta.error(format!("invalid columnar trait path: {e}")))?;
                    columnar_trait_path_ts = quote! { #path };
                    Ok(())
                } else {
                    Err(meta.error("unsupported key in #[df_derive(...)] attribute"))
                }
            });
            if let Err(err) = parse_res {
                return err.to_compile_error().into();
            }
        }
    }
    let config = codegen::MacroConfig {
        to_dataframe_trait_path: to_df_trait_path_ts,
        columnar_trait_path: columnar_trait_path_ts,
    };
    // Build the intermediate representation
    let ir = match parser::parse_to_ir(&ast) {
        Ok(ir) => ir,
        Err(e) => return e.to_compile_error().into(),
    };

    // Delegate to the codegen orchestrator
    let generated = codegen::generate_code(&ir, &config);
    TokenStream::from(generated)
}
