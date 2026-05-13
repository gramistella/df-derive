mod columnar_impl;
mod encoder;
mod external_paths;
mod helpers;
mod nested;
mod strategy;
mod trait_impl;
mod type_registry;

use crate::ir::{DisplayBase, LeafSpec, StringyBase, StructIR, TupleElement};
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};

/// Macro-wide configuration for generated code
#[allow(clippy::struct_field_names)]
pub struct MacroConfig {
    /// Fully-qualified path to the `ToDataFrame` trait (e.g., `df_derive::dataframe::ToDataFrame`)
    pub to_dataframe_trait_path: TokenStream,
    /// Fully-qualified path to the Columnar trait (e.g., `df_derive::dataframe::Columnar`)
    pub columnar_trait_path: TokenStream,
    /// Fully-qualified path to the `Decimal128Encode` trait used by Decimal
    /// fields to dispatch the value-to-i128-mantissa rescale through
    /// user-controlled code. Defaults to a sibling of the `ToDataFrame` trait
    /// (`<default-or-user-mod>::Decimal128Encode`); custom decimal backends
    /// override this with `#[df_derive(decimal128_encode = "...")]`.
    pub decimal128_encode_trait_path: TokenStream,
}

fn resolve_dataframe_mod_for_crate(name: &str, itself: TokenStream) -> Option<TokenStream> {
    match crate_name(name) {
        Ok(FoundCrate::Name(resolved)) => {
            let ident = format_ident!("{}", resolved);
            Some(quote! { ::#ident::dataframe })
        }
        Ok(FoundCrate::Itself) => Some(itself),
        Err(_) => None,
    }
}

pub fn resolve_default_dataframe_mod() -> TokenStream {
    // Default discovery order:
    // - `df-derive` facade (`df_derive::dataframe`)
    // - `df-derive-core` shared runtime (`df_derive_core::dataframe`)
    // - `paft-utils` direct runtime (`paft_utils::dataframe`)
    // - `paft` facade (`paft::dataframe`)
    // - legacy local fallback (`crate::core::dataframe`)
    resolve_dataframe_mod_for_crate("df-derive", quote! { ::df_derive::dataframe })
        .or_else(|| resolve_dataframe_mod_for_crate("df-derive-core", quote! { crate::dataframe }))
        .or_else(|| resolve_dataframe_mod_for_crate("paft-utils", quote! { crate::dataframe }))
        .or_else(|| resolve_dataframe_mod_for_crate("paft", quote! { crate::dataframe }))
        .unwrap_or_else(|| quote! { crate::core::dataframe })
}

pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let trait_impl = trait_impl::generate_trait_impl(ir, config);
    let columnar_impl = columnar_impl::generate_columnar_impl(ir, config);
    let eager_asserts = helpers::generate_eager_asserts(ir);
    let pp = external_paths::prelude();
    let assemble_helper = encoder::idents::assemble_helper();

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
    // every nested-`Vec` bulk emitter site.
    quote! {
        const _: () = {
            #eager_asserts

            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn #assemble_helper(
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
        };
    }
}

#[derive(Default)]
struct GenericRequirements {
    nested_params: Vec<syn::Ident>,
    nested_paths: Vec<syn::Path>,
    decimal_params: Vec<syn::Ident>,
    decimal_paths: Vec<syn::Path>,
    as_ref_str: Vec<syn::Ident>,
    as_ref_str_paths: Vec<syn::Path>,
    display_params: Vec<syn::Ident>,
    display_paths: Vec<syn::Path>,
}

fn push_unique(out: &mut Vec<syn::Ident>, ident: &syn::Ident) {
    if !out.iter().any(|existing| existing == ident) {
        out.push(ident.clone());
    }
}

fn contains_ident(items: &[syn::Ident], ident: &syn::Ident) -> bool {
    items.iter().any(|item| item == ident)
}

fn push_unique_path(out: &mut Vec<syn::Path>, path: &syn::Path) {
    let key = path.to_token_stream().to_string();
    if !out
        .iter()
        .any(|existing| existing.to_token_stream().to_string() == key)
    {
        out.push(path.clone());
    }
}

fn collect_tuple_element_requirements(elem: &TupleElement, reqs: &mut GenericRequirements) {
    collect_leaf_requirements(&elem.leaf_spec, reqs);
}

fn collect_leaf_requirements(leaf: &LeafSpec, reqs: &mut GenericRequirements) {
    match leaf {
        LeafSpec::Generic(ident) => {
            push_unique(&mut reqs.nested_params, ident);
        }
        LeafSpec::Struct(path) => {
            push_unique_path(&mut reqs.nested_paths, path);
        }
        LeafSpec::AsStr(StringyBase::Generic(ident)) => {
            push_unique(&mut reqs.as_ref_str, ident);
        }
        LeafSpec::AsStr(StringyBase::Struct(path)) => {
            push_unique_path(&mut reqs.as_ref_str_paths, path);
        }
        LeafSpec::AsString(DisplayBase::Generic(ident)) => {
            push_unique(&mut reqs.display_params, ident);
        }
        LeafSpec::AsString(DisplayBase::Struct(path)) => {
            push_unique_path(&mut reqs.display_paths, path);
        }
        LeafSpec::Tuple(elements) => {
            for elem in elements {
                collect_tuple_element_requirements(elem, reqs);
            }
        }
        LeafSpec::Numeric(_)
        | LeafSpec::String
        | LeafSpec::Bool
        | LeafSpec::DateTime(_)
        | LeafSpec::NaiveDateTime(_)
        | LeafSpec::NaiveDate
        | LeafSpec::NaiveTime
        | LeafSpec::Duration { .. }
        | LeafSpec::Decimal { .. }
        | LeafSpec::AsString(_)
        | LeafSpec::AsStr(_)
        | LeafSpec::Binary => {}
    }
}

fn collect_generic_requirements(ir: &StructIR) -> GenericRequirements {
    let mut reqs = GenericRequirements::default();

    for field in &ir.fields {
        collect_leaf_requirements(&field.leaf_spec, &mut reqs);

        for ident in &field.decimal_generic_params {
            push_unique(&mut reqs.decimal_params, ident);
        }

        if let Some(path) = &field.decimal_backend_path {
            push_unique_path(&mut reqs.decimal_paths, path);
        }
    }

    reqs
}

/// Build `impl_generics`, `ty_generics`, and `where_clause` token streams
/// suitable for splicing into an `impl` header. Generic bounds are driven by
/// each parameter's role: direct generic dataframe payloads need `ToDataFrame
/// + Columnar`, decimal backends need `Decimal128Encode`, generic `as_str`
/// leaves need `AsRef<str>`, generic `as_string` leaves need `Display`, and
/// concrete conversion/nested paths receive exact `where` predicates.
pub fn impl_parts_with_bounds(
    ir: &StructIR,
    config: &MacroConfig,
) -> (TokenStream, TokenStream, TokenStream) {
    let mut generics = ir.generics.clone();
    let reqs = collect_generic_requirements(ir);

    let to_df_trait = &config.to_dataframe_trait_path;
    let columnar_trait = &config.columnar_trait_path;
    let decimal_trait = &config.decimal128_encode_trait_path;
    let to_df_bound: syn::TypeParamBound =
        syn::parse2(quote! { #to_df_trait }).expect("trait path should parse as bound");
    let columnar_bound: syn::TypeParamBound =
        syn::parse2(quote! { #columnar_trait }).expect("trait path should parse as bound");
    let decimal_bound: syn::TypeParamBound =
        syn::parse2(quote! { #decimal_trait }).expect("trait path should parse as bound");
    let as_ref_str_bound: syn::TypeParamBound = syn::parse2(quote! { ::core::convert::AsRef<str> })
        .expect("AsRef<str> should parse as bound");
    let display_bound: syn::TypeParamBound =
        syn::parse2(quote! { ::core::fmt::Display }).expect("Display should parse as bound");
    // No `Clone` bound: bulk emitters collect `Vec<&T>` and route through
    // `Columnar::columnar_from_refs`, and every primitive-vec branch in the
    // encoder IR borrows from the for-loop binding directly. A user with a
    // non-`Clone` payload (e.g. `T: ToDataFrame + Columnar` only) can derive
    // `ToDataFrame` on a struct holding `T` without that bound leaking from
    // the macro.
    for tp in generics.type_params_mut() {
        if contains_ident(&reqs.nested_params, &tp.ident) {
            tp.bounds.push(to_df_bound.clone());
            tp.bounds.push(columnar_bound.clone());
        }

        if contains_ident(&reqs.decimal_params, &tp.ident) {
            tp.bounds.push(decimal_bound.clone());
        }

        if contains_ident(&reqs.as_ref_str, &tp.ident) {
            tp.bounds.push(as_ref_str_bound.clone());
        }

        if contains_ident(&reqs.display_params, &tp.ident) {
            tp.bounds.push(display_bound.clone());
        }
    }

    if !reqs.nested_paths.is_empty()
        || !reqs.as_ref_str_paths.is_empty()
        || !reqs.display_paths.is_empty()
        || !reqs.decimal_paths.is_empty()
    {
        let where_clause_mut = generics.make_where_clause();

        for path in &reqs.nested_paths {
            let nested_ty = quote! { #path };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #nested_ty: #to_df_trait })
                    .expect("nested ToDataFrame where predicate should parse"),
            );
            where_clause_mut.predicates.push(
                syn::parse2(quote! { #nested_ty: #columnar_trait })
                    .expect("nested Columnar where predicate should parse"),
            );
        }
        for path in &reqs.as_ref_str_paths {
            let as_str_ty = quote! { #path };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #as_str_ty: ::core::convert::AsRef<str> })
                    .expect("as_str path where predicate should parse"),
            );
        }
        for path in &reqs.display_paths {
            let display_ty = quote! { #path };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #display_ty: ::core::fmt::Display })
                    .expect("as_string path where predicate should parse"),
            );
        }
        for path in &reqs.decimal_paths {
            let decimal_ty = quote! { #path };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #decimal_ty: #decimal_trait })
                    .expect("decimal backend where predicate should parse"),
            );
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    (
        quote! { #impl_generics },
        quote! { #ty_generics },
        quote! { #where_clause },
    )
}
