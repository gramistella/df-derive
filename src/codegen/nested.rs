// Codegen for fields whose base type is a derived struct or a generic type
// parameter routed through `ToDataFrame` / `Columnar`. Two flavors live here:
//
//   - Concrete-struct (`is_generic == false`) calls the inherent
//     `__df_derive_vec_to_inner_list_values` helper for nested-Vec
//     aggregation.
//   - Generic-parameter (`is_generic == true`) uses only the `ToDataFrame`
//     and `Columnar` traits — nothing inherent — because the parameter type
//     isn't known at macro-expansion time.
//
// The two flavors share `gen_recursive_per_element_to_list_anyvalues` for
// deeper-nested vec shapes that fall outside the typed fast paths.
//
// Bare leaf, `Option<Inner>`, and `Vec<Inner>` shapes never reach this
// module's per-row push code — they're served by bulk emitters in
// `bulk.rs`. The on-leaf branch in `generate_nested_for_columnar_push`
// only fires for the rare `Option<Option<Inner>>`-style shape.

use crate::ir::{Wrapper, has_vec, vec_count};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::populator_idents::PopulatorIdents;

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
        let mut #nn_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        for __df_derive_maybe in (#acc).iter() {
            match __df_derive_maybe {
                ::std::option::Option::Some(v) => {
                    #pos_ident.push(::std::option::Option::Some(
                        #nn_ident.len() as #pp::IdxSize,
                    ));
                    #nn_ident.push((*v).clone());
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
            let #vals_ident = #ty::__df_derive_vec_to_inner_list_values(&#nn_ident)?;
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
                        ComputeError: "df-derive: expected list AnyValue from __df_derive_vec_to_inner_list_values (codegen invariant violation)"
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

// --- Context-specific generators ---

pub fn generate_nested_for_columnar_push(
    type_path: &TokenStream,
    access: &TokenStream,
    wrappers: &[Wrapper],
    idx: usize,
    is_generic: bool,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let ty = type_path.clone();
    let vec = has_vec(wrappers);

    // For non-vec shapes the populator is `Vec<Vec<AnyValue>>` (one inner
    // Vec per inner schema column, accumulating one AnyValue per outer row).
    // For vec shapes the populator is `Vec<Box<dyn ListBuilderTrait>>`
    // (one builder per inner schema column, accumulating one outer-list
    // entry per outer row). The on-leaf branch only runs for non-vec
    // shapes — `process_wrappers` reaches the leaf only when no `Vec`
    // wrapper is present.
    //
    // After unification, this on-leaf is only reached for the rare
    // `Option<Option<T>>`-style shape (since `[]`, `[Option]`, and `[Vec]`
    // are now all bulk-emitted). Both concrete and generic branches call
    // `to_inner_values(&self)` on the `ToDataFrame` trait — for the derive's
    // own impls, that's the optimized override that pushes column values
    // directly; for foreign impls, it's the trait's default that round-trips
    // through `to_dataframe()`.
    let cols_ident = PopulatorIdents::nested_struct_cols(idx);
    let lbs_ident = PopulatorIdents::nested_list_builders(idx);

    let on_leaf = |acc: &TokenStream| {
        let cols_ident = cols_ident.clone();
        quote! {
            let __df_derive_vs = (#acc).to_inner_values()?;
            for (j, __df_derive_v) in __df_derive_vs.into_iter().enumerate() {
                #cols_ident[j].push(__df_derive_v);
            }
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        if vec {
            let lbs_ident = lbs_ident.clone();
            quote! {
                for j in 0..#lbs_ident.len() {
                    #pp::ListBuilderTrait::append_null(&mut *#lbs_ident[j]);
                }
            }
        } else {
            let cols_ident = cols_ident.clone();
            quote! {
                for j in 0..#cols_ident.len() {
                    #cols_ident[j].push(#pp::AnyValue::Null);
                }
            }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let lbs_ident = lbs_ident.clone();
        let list_vals_ts = if is_generic {
            gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
        } else {
            gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
        };
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<#pp::AnyValue> = { #list_vals_ts };
            for (j, __df_derive_v) in __df_derive_vals.into_iter().enumerate() {
                match __df_derive_v {
                    #pp::AnyValue::List(__df_derive_inner) => {
                        #pp::ListBuilderTrait::append_series(
                            &mut *#lbs_ident[j],
                            &__df_derive_inner,
                        )?;
                    }
                    #pp::AnyValue::Null => {
                        #pp::ListBuilderTrait::append_null(&mut *#lbs_ident[j]);
                    }
                    _ => {
                        return ::std::result::Result::Err(#pp::polars_err!(
                            ComputeError: "df-derive: expected list or null AnyValue from nested vec helper (codegen invariant violation)"
                        ));
                    }
                }
            }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
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
    generate_empty_series_for_struct(type_path, name, has_vec(wrappers))
}

pub fn nested_decls(wrappers: &[Wrapper], type_path: &TokenStream, idx: usize) -> Vec<TokenStream> {
    let pp = super::polars_paths::prelude();
    let mut decls: Vec<TokenStream> = Vec::new();
    let vec = has_vec(wrappers);
    if vec {
        // Vec<Struct> shapes that didn't take the bulk-concrete fast path
        // (i.e. `Vec<Option<Struct>>`, `Vec<Vec<Struct>>`, etc.). Per parent
        // row we still call the inner helper to get one inner Series per
        // schema column; instead of accumulating those into a
        // `Vec<AnyValue::List>` and rebuilding the outer list series via
        // `Series::new`, we feed each inner Series straight into a typed
        // `ListBuilder` per inner column. Skips the AnyValue inference scan
        // and per-row `cast(inner_type)` Polars does inside
        // `any_values_to_list`.
        //
        // For nested-Vec shapes (`Vec<Vec<Struct>>`, …), the per-row inner
        // Series feeding the builder is itself `List<…>`-shaped, so the
        // builder's inner dtype must include `(vec_count - 1)` extra
        // `List<>` layers around the inner-struct schema dtype — see the
        // analogous wrap in `outer_list_inner_dtype` for primitives. Without
        // this, `ListPrimitiveChunkedBuilder` rejects the deeper `list[…]`
        // slice with a `SchemaMismatch`.
        let schema_ident = PopulatorIdents::nested_vec_schema(idx);
        let lbs_ident = PopulatorIdents::nested_list_builders(idx);
        let cab = super::polars_paths::chunked_array_builder();
        let wrap_extra = gen_wrap_dtype_layers(vec_count(wrappers).saturating_sub(1));
        decls.push(quote! { let #schema_ident = #type_path::schema()?; });
        decls.push(quote! {
            let mut #lbs_ident: ::std::vec::Vec<
                ::std::boxed::Box<dyn #pp::ListBuilderTrait>,
            > = #schema_ident
                .iter()
                .map(|(_, __df_derive_inner_dtype)| {
                    let mut __df_derive_wrapped = __df_derive_inner_dtype.clone();
                    #wrap_extra
                    #cab::get_list_builder(
                        &__df_derive_wrapped,
                        items.len() * 4,
                        items.len(),
                        "".into(),
                    )
                })
                .collect();
        });
    } else {
        let schema_ident = PopulatorIdents::nested_struct_schema(idx);
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        decls.push(quote! { let #schema_ident = #type_path::schema()?; });
        decls.push(quote! {
            let mut #cols_ident: ::std::vec::Vec<::std::vec::Vec<#pp::AnyValue>> =
                #schema_ident
                    .iter()
                    .map(|_| ::std::vec::Vec::with_capacity(items.len()))
                    .collect();
        });
    }
    decls
}

pub fn nested_finishers_for_vec_anyvalues(wrappers: &[Wrapper], idx: usize) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let vec = has_vec(wrappers);
    let schema_ident = if vec {
        PopulatorIdents::nested_vec_schema(idx)
    } else {
        PopulatorIdents::nested_struct_schema(idx)
    };
    if vec {
        let lbs_ident = PopulatorIdents::nested_list_builders(idx);
        quote! {
            for j in 0..#schema_ident.len() {
                let inner = #pp::IntoSeries::into_series(
                    #pp::ListBuilderTrait::finish(&mut *#lbs_ident[j]),
                );
                out_values.push(#pp::AnyValue::List(inner));
            }
        }
    } else {
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        quote! {
            for j in 0..#schema_ident.len() {
                let inner = <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &#cols_ident[j]);
                out_values.push(#pp::AnyValue::List(inner));
            }
        }
    }
}

pub fn nested_columnar_builders(
    wrappers: &[Wrapper],
    idx: usize,
    field_name: &str,
) -> Vec<TokenStream> {
    let pp = super::polars_paths::prelude();
    let vec = has_vec(wrappers);
    let schema_ident = if vec {
        PopulatorIdents::nested_vec_schema(idx)
    } else {
        PopulatorIdents::nested_struct_schema(idx)
    };
    let name = field_name;
    if vec {
        let lbs_ident = PopulatorIdents::nested_list_builders(idx);
        vec![quote! {{
            for (j, (col_name, _)) in #schema_ident.iter().enumerate() {
                let full_name = ::std::format!("{}.{}", #name, col_name);
                let s = #pp::IntoSeries::into_series(
                    #pp::ListBuilderTrait::finish(&mut *#lbs_ident[j]),
                )
                .with_name(full_name.as_str().into());
                columns.push(s.into());
            }
        }}]
    } else {
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        vec![quote! {{
            for (j, (col_name, _)) in #schema_ident.iter().enumerate() {
                let full_name = ::std::format!("{}.{}", #name, col_name);
                let s = <#pp::Series as #pp::NamedFrom<_, _>>::new(full_name.as_str().into(), &#cols_ident[j]);
                columns.push(s.into());
            }
        }}]
    }
}

// --- Schema and series-shape helpers ---

pub fn generate_schema_entries_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    quote! {
        {
            let mut nested_fields: ::std::vec::Vec<(::std::string::String, #pp::DataType)> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = ::std::format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    #pp::DataType::List(::std::boxed::Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                nested_fields.push((prefixed_name, dtype));
            }
            nested_fields
        }
    }
}

fn generate_empty_series_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    quote! {
        {
            let mut nested_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = ::std::format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    #pp::DataType::List(::std::boxed::Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                let empty_series = #pp::Series::new_empty(prefixed_name.as_str().into(), &dtype);
                nested_series.push(empty_series.into());
            }
            nested_series
        }
    }
}
