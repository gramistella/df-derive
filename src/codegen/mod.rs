mod columnar_impl;
mod common;
mod encoder;
mod helpers;
mod nested;
mod polars_paths;
mod populator_idents;
mod primitive;
mod strategy;
mod trait_impl;
mod type_registry;
mod wrapper_processor;

use crate::ir::StructIR;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Macro-wide configuration for generated code
#[allow(clippy::struct_field_names)]
pub struct MacroConfig {
    /// Fully-qualified path to the `ToDataFrame` trait (e.g., `paft::dataframe::ToDataFrame`)
    pub to_dataframe_trait_path: TokenStream,
    /// Fully-qualified path to the Columnar trait (e.g., `paft::dataframe::Columnar`)
    pub columnar_trait_path: TokenStream,
    /// Fully-qualified path to the `Decimal128Encode` trait used by Decimal
    /// fields to dispatch the value-to-i128-mantissa rescale through
    /// user-controlled code. Defaults to a sibling of the `ToDataFrame` trait
    /// (`<default-or-user-mod>::Decimal128Encode`); custom decimal backends
    /// override this with `#[df_derive(decimal128_encode = "...")]`.
    pub decimal128_encode_trait_path: TokenStream,
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
    let pp = polars_paths::prelude();

    // Wrap the entire generated impl set in a per-derive `const _: () = { ... };`
    // scope. Inherent impls inside an anonymous const still apply to the outer
    // type (the same trick `serde_derive` uses), so `Foo::__df_derive_*` calls
    // from sibling derives keep working. The free helper
    // `__df_derive_assemble_list_series_unchecked` lives inside the same scope,
    // hidden from the user's namespace, and is the only place the bulk path's
    // `unsafe` call to `Series::from_chunks_and_dtype_unchecked` lives — so
    // `clippy::unsafe_derive_deserialize` no longer sees `unsafe` inside any
    // impl method on `Self` and stops firing on downstream
    // `#[derive(ToDataFrame, Deserialize)]` types.
    //
    // The helper is `#[inline(always)]` so the call site collapses; the
    // bulk path's perf is unchanged. Plain `#[inline]` was insufficient on
    // `bench/02_empty_nested_vec` (~7% regression), where the per-row work
    // is dominated by the `Series::from_chunks_and_dtype_unchecked` call
    // and a non-collapsed call frame is observable. Its signature is fully
    // concrete (no generics) so a single instantiation per derive serves
    // both `gen_bulk_vec` and `gen_bulk_option_vec` call sites.
    quote! {
        const _: () = {
            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn __df_derive_assemble_list_series_unchecked(
                list_arr: #pp::LargeListArray,
                inner_logical_dtype: #pp::DataType,
            ) -> #pp::Series {
                // SAFETY: `list_arr` was constructed with
                // `LargeListArray::default_datatype(inner_arrow_dtype)`, where
                // `inner_arrow_dtype` is the arrow dtype of the inner Series
                // chunk produced by `Inner::columnar_from_refs`. That arrow
                // dtype is the physical projection of `inner_logical_dtype`
                // (both derive from the same `Inner::schema()` entry), so
                // the chunk's arrow dtype matches the
                // `List(inner_logical_dtype)` declaration. Validity (when
                // present) was built with one bit per outer row, satisfying
                // `LargeListArray::new`.
                unsafe {
                    #pp::Series::from_chunks_and_dtype_unchecked(
                        "".into(),
                        ::std::vec![
                            ::std::boxed::Box::new(list_arr) as #pp::ArrayRef,
                        ],
                        &#pp::DataType::List(::std::boxed::Box::new(inner_logical_dtype)),
                    )
                }
            }

            #trait_impl
            #columnar_impl
            #helpers_impl
        };
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
    // No `Clone` bound: bulk emitters collect `Vec<&T>` and route through
    // `Columnar::columnar_from_refs`, and the only primitive-vec branch that
    // previously cloned (`gen_primitive_vec_inner_series`'s deep-fallback)
    // now borrows from the for-loop binding directly. A user with a
    // non-`Clone` payload (e.g. `T: ToDataFrame + Columnar` only) can derive
    // `ToDataFrame` on a struct holding `T` without that bound leaking from
    // the macro.
    for tp in generics.type_params_mut() {
        tp.bounds.push(to_df_bound.clone());
        tp.bounds.push(columnar_bound.clone());
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    (
        quote! { #impl_generics },
        quote! { #ty_generics },
        quote! { #where_clause },
    )
}
