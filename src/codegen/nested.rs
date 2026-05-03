// Per-row Codegen scaffolding kept for the `to_inner_values(&self)` trait
// override. After Step 4 every nested-struct/generic columnar/vec-anyvalues
// path routes through the encoder fold; only the single-instance
// `gen_for_anyvalue` driver still walks the wrapper stack here, plus the
// schema/empty-frame helpers used by `to_dataframe::schema` /
// `empty_dataframe`.

use crate::ir::{Wrapper, vec_count};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

/// Emit a runtime loop that wraps `__df_derive_wrapped: DataType` in `layers`
/// `List<>` envelopes. Returns an empty token stream when `layers == 0` so the
/// caller does not emit `for _ in 0..0`, which trips
/// `clippy::reversed_empty_ranges` inside the user's expanded code.
fn gen_wrap_dtype_layers(layers: usize) -> TokenStream {
    if layers == 0 {
        TokenStream::new()
    } else {
        let pp = super::polars_paths::prelude();
        quote! {
            for _ in 0..#layers {
                __df_derive_wrapped = #pp::DataType::List(
                    ::std::boxed::Box::new(__df_derive_wrapped),
                );
            }
        }
    }
}

// --- Vec<Struct> → Vec<AnyValue::List> dispatchers ---

/// Build tokens that evaluate to `Vec<polars::prelude::AnyValue>`, where each
/// element is `AnyValue::List(inner_series)` for one inner schema column of
/// `ty` aggregated across the outer `Vec`. Dispatches to one of three
/// implementations based on the wrapper shape inside the outer Vec:
///
/// - `Vec<Struct>` (tail empty): inherent fast path on the inner type.
/// - `Vec<Option<Struct>>` (tail = `[Option]`): typed `Series::take` scatter,
///   no per-row `AnyValue` round-trip.
/// - Deeper nestings (`Vec<Vec<Struct>>` etc.): per-element `AnyValue`
///   round-trip aggregated into typed inner Series.
fn gen_nested_vec_to_list_anyvalues(
    ty: &TokenStream,
    acc: &TokenStream,
    tail: &[Wrapper],
) -> TokenStream {
    match tail {
        [] => gen_nested_vec_anyvalues_flat(ty, acc),
        [Wrapper::Option] => gen_nested_vec_anyvalues_option(ty, acc),
        _ => gen_recursive_per_element_to_list_anyvalues(ty, acc, tail, |elem, vals| {
            generate_nested_for_anyvalue(ty, vals, &quote! { #elem }, tail, false)
        }),
    }
}

/// Fast path for `Vec<Struct>`: defer to the inherent helper on the inner
/// type, which already produces the typed `AnyValue::List` cells without any
/// per-row round-trip.
fn gen_nested_vec_anyvalues_flat(ty: &TokenStream, acc: &TokenStream) -> TokenStream {
    quote! { #ty::__df_derive_vec_to_inner_list_values(&(#acc))? }
}

/// Semi-optimized path for `Vec<Option<Struct>>`. Builds the inner `DataFrame`
/// once over the non-null subset, then per inner column scatters back over the
/// original positions via `Series::take(&IdxCa)`. Typed Series in, typed
/// Series out — no `Vec<AnyValue>` round-trip (which previously paid for an
/// `AnyValue` dispatch per outer position plus an inferring scan when the outer
/// Series was rebuilt).
///
/// Borrowing path: collects `&Struct` references for each `Some(v)` and feeds
/// them into `__df_derive_refs_to_inner_list_values`, avoiding the per-Some
/// clone that an owned `Vec<Self>` would require. Critical when the inner
/// struct holds `String`s or other heap-allocating fields.
fn gen_nested_vec_anyvalues_option(ty: &TokenStream, acc: &TokenStream) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
    let pos_ident = syn::Ident::new("__df_derive_pos", proc_macro2::Span::call_site());
    let nn_ident = syn::Ident::new("__df_derive_nn", proc_macro2::Span::call_site());
    let vals_ident = syn::Ident::new("__df_derive_vals", proc_macro2::Span::call_site());
    let take_ident = syn::Ident::new("__df_derive_take", proc_macro2::Span::call_site());
    quote! {{
        let #schema_ident = #ty::schema()?;
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity((#acc).len());
        let mut #nn_ident: ::std::vec::Vec<&#ty> = ::std::vec::Vec::new();
        for __df_derive_maybe in (#acc).iter() {
            match __df_derive_maybe {
                ::std::option::Option::Some(v) => {
                    #pos_ident.push(::std::option::Option::Some(
                        #nn_ident.len() as #pp::IdxSize,
                    ));
                    #nn_ident.push(v);
                }
                ::std::option::Option::None => #pos_ident.push(::std::option::Option::None),
            }
        }
        if #nn_ident.is_empty() {
            // All-None path: produce one outer-list cell per inner schema
            // column, each a typed-null Series of length `(#acc).len()`.
            // Pre-typing avoids feeding `dtype Null` into a list builder
            // that expects e.g. `list[Float64]` (which
            // `ListPrimitiveChunkedBuilder::append_series` rejects).
            let mut __df_derive_out: ::std::vec::Vec<#pp::AnyValue> =
                ::std::vec::Vec::with_capacity(#schema_ident.len());
            for (_inner_name, __df_derive_inner_dtype) in #schema_ident.iter() {
                let inner = #pp::Series::new_empty("".into(), __df_derive_inner_dtype)
                    .extend_constant(#pp::AnyValue::Null, (#acc).len())?;
                __df_derive_out.push(#pp::AnyValue::List(inner));
            }
            __df_derive_out
        } else {
            let #vals_ident = #ty::__df_derive_refs_to_inner_list_values(&#nn_ident)?;
            let #take_ident: #pp::IdxCa =
                <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #pos_ident.iter().copied(),
                );
            let mut __df_derive_out: ::std::vec::Vec<#pp::AnyValue> =
                ::std::vec::Vec::with_capacity(#schema_ident.len());
            for j in 0..#schema_ident.len() {
                let inner = match &#vals_ident[j] {
                    #pp::AnyValue::List(__df_derive_inner_full) => {
                        __df_derive_inner_full.take(&#take_ident)?
                    }
                    _ => return ::std::result::Result::Err(#pp::polars_err!(
                        ComputeError: "df-derive: expected list AnyValue from __df_derive_refs_to_inner_list_values (codegen invariant violation)"
                    )),
                };
                __df_derive_out.push(#pp::AnyValue::List(inner));
            }
            __df_derive_out
        }
    }}
}

/// Recursive per-element machinery shared by the deeper nested-vec shapes and
/// the trait-only generic-vec path.
///
/// Per outer element, `build_recur_elem(elem, vals)` is invoked once to
/// produce the token stream that pushes one `AnyValue` per inner schema column
/// onto the runtime `vals` accumulator. Those values are scattered into a
/// per-column buffer that becomes the typed inner Series wrapped in
/// `AnyValue::List`.
///
/// Each `Vec` in `tail` adds one `List<>` layer to the inferred dtype of the
/// inner Series rebuilt below; the empty-input branch wraps the schema-declared
/// inner dtype the same number of times so a downstream typed list builder
/// doesn't reject the empty Series with `SchemaMismatch`.
fn gen_recursive_per_element_to_list_anyvalues<F>(
    ty: &TokenStream,
    acc: &TokenStream,
    tail: &[Wrapper],
    build_recur_elem: F,
) -> TokenStream
where
    F: FnOnce(&Ident, &Ident) -> TokenStream,
{
    let pp = super::polars_paths::prelude();
    let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
    let cols_buf_ident = syn::Ident::new("__df_derive_cols_buf", proc_macro2::Span::call_site());
    let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
    let per_item_vals_ident =
        syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());
    let recur_elem_ts = build_recur_elem(&elem_ident, &per_item_vals_ident);
    let wrap_extra_for_empty = gen_wrap_dtype_layers(vec_count(tail));

    quote! {{
        let #schema_ident = #ty::schema()?;
        let mut #cols_buf_ident: ::std::vec::Vec<::std::vec::Vec<#pp::AnyValue>> =
            #schema_ident.iter().map(|_| ::std::vec::Vec::with_capacity((#acc).len())).collect();
        for #elem_ident in (#acc).iter() {
            let mut #per_item_vals_ident: ::std::vec::Vec<#pp::AnyValue> = ::std::vec::Vec::new();
            { #recur_elem_ts }
            for (j, v) in #per_item_vals_ident.into_iter().enumerate() { #cols_buf_ident[j].push(v); }
        }
        let mut __df_derive_out: ::std::vec::Vec<#pp::AnyValue> = ::std::vec::Vec::with_capacity(#schema_ident.len());
        for (j, (_inner_name, __df_derive_inner_dtype)) in #schema_ident.iter().enumerate() {
            let inner = if #cols_buf_ident[j].is_empty() {
                let mut __df_derive_wrapped = __df_derive_inner_dtype.clone();
                #wrap_extra_for_empty
                #pp::Series::new_empty("".into(), &__df_derive_wrapped)
            } else {
                <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &#cols_buf_ident[j])
            };
            __df_derive_out.push(#pp::AnyValue::List(inner));
        }
        __df_derive_out
    }}
}

/// Trait-only equivalent of `gen_nested_vec_to_list_anyvalues` for fields whose
/// base type is a generic type parameter. Avoids any inherent helpers and uses
/// only `ToDataFrame` / `Columnar` trait methods. Always recursive — there's
/// no inherent fast path available behind a trait bound.
fn gen_generic_vec_to_list_anyvalues(
    ty: &TokenStream,
    acc: &TokenStream,
    tail: &[Wrapper],
) -> TokenStream {
    gen_recursive_per_element_to_list_anyvalues(ty, acc, tail, |elem, vals| {
        generate_nested_for_anyvalue(ty, vals, &quote! { #elem }, tail, true)
    })
}

pub fn generate_nested_for_anyvalue(
    type_path: &TokenStream,
    values_vec_ident: &Ident,
    access: &TokenStream,
    wrappers: &[Wrapper],
    is_generic: bool,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| {
        // Both concrete and generic branches call `to_inner_values(&self)` on
        // the `ToDataFrame` trait. The derive's optimized override avoids the
        // one-row `DataFrame` round-trip; foreign impls still get the trait's
        // default that goes through `to_dataframe()`.
        quote! {
            let __df_derive_vs = (#acc).to_inner_values()?;
            #values_vec_ident.extend(__df_derive_vs);
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        quote! {
            let schema = #ty::schema()?;
            for _ in 0..schema.len() { #values_vec_ident.push(#pp::AnyValue::Null); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let list_vals_ts = if is_generic {
            gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
        } else {
            gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
        };
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<#pp::AnyValue> = { #list_vals_ts };
            for v in __df_derive_vals.into_iter() { #values_vec_ident.push(v); }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

// --- Columnar populator decls and finishers ---

pub fn nested_empty_series_row(
    type_path: &TokenStream,
    name: &str,
    wrappers: &[Wrapper],
) -> TokenStream {
    generate_empty_series_for_struct(type_path, name, vec_count(wrappers))
}

// --- Schema and series-shape helpers ---

pub fn generate_schema_entries_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    list_layers: usize,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let wrap_layers = gen_wrap_dtype_layers(list_layers);
    quote! {
        {
            let mut nested_fields: ::std::vec::Vec<(::std::string::String, #pp::DataType)> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = ::std::format!("{}.{}", #column_name, inner_name);
                let mut __df_derive_wrapped: #pp::DataType = inner_dtype;
                #wrap_layers
                nested_fields.push((prefixed_name, __df_derive_wrapped));
            }
            nested_fields
        }
    }
}

fn generate_empty_series_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    list_layers: usize,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let wrap_layers = gen_wrap_dtype_layers(list_layers);
    quote! {
        {
            let mut nested_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = ::std::format!("{}.{}", #column_name, inner_name);
                let mut __df_derive_wrapped: #pp::DataType = inner_dtype;
                #wrap_layers
                let empty_series = #pp::Series::new_empty(prefixed_name.as_str().into(), &__df_derive_wrapped);
                nested_series.push(empty_series.into());
            }
            nested_series
        }
    }
}
