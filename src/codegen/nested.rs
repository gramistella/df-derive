// Codegen for fields whose base type is a derived struct or a generic type
// parameter routed through `ToDataFrame` / `Columnar`. Two parallel families
// of generators live here:
//
//   - The concrete-struct path (`is_generic == false`) calls inherent helpers
//     emitted on the user's struct (`__df_derive_vec_to_inner_list_values`,
//     `__df_derive_to_anyvalues`, `__df_derive_collect_vec_as_prefixed_list_series`).
//   - The generic-parameter path (`is_generic == true`) uses only the
//     `ToDataFrame` and `Columnar` traits — nothing inherent — because the
//     parameter type isn't known at macro-expansion time.
//
// The two paths share `gen_recursive_per_element_to_list_anyvalues` for the
// deeper-nested vec shapes that fall outside the typed fast paths.

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
        quote! {
            for _ in 0..#layers {
                __df_derive_wrapped = polars::prelude::DataType::List(
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
    let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
    let pos_ident = syn::Ident::new("__df_derive_pos", proc_macro2::Span::call_site());
    let nn_ident = syn::Ident::new("__df_derive_nn", proc_macro2::Span::call_site());
    let vals_ident = syn::Ident::new("__df_derive_vals", proc_macro2::Span::call_site());
    let take_ident = syn::Ident::new("__df_derive_take", proc_macro2::Span::call_site());
    quote! {{
        let #schema_ident = #ty::schema()?;
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<polars::prelude::IdxSize>> =
            ::std::vec::Vec::with_capacity((#acc).len());
        let mut #nn_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        for __df_derive_maybe in (#acc).iter() {
            match __df_derive_maybe {
                ::std::option::Option::Some(v) => {
                    #pos_ident.push(::std::option::Option::Some(
                        #nn_ident.len() as polars::prelude::IdxSize,
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
            let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> =
                ::std::vec::Vec::with_capacity(#schema_ident.len());
            for (_inner_name, __df_derive_inner_dtype) in #schema_ident.iter() {
                let inner = polars::prelude::Series::new_empty("".into(), __df_derive_inner_dtype)
                    .extend_constant(polars::prelude::AnyValue::Null, (#acc).len())?;
                __df_derive_out.push(polars::prelude::AnyValue::List(inner));
            }
            __df_derive_out
        } else {
            let #vals_ident = #ty::__df_derive_vec_to_inner_list_values(&#nn_ident)?;
            let #take_ident: polars::prelude::IdxCa =
                <polars::prelude::IdxCa as polars::prelude::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #pos_ident.iter().copied(),
                );
            let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> =
                ::std::vec::Vec::with_capacity(#schema_ident.len());
            for j in 0..#schema_ident.len() {
                let inner = match &#vals_ident[j] {
                    polars::prelude::AnyValue::List(__df_derive_inner_full) => {
                        __df_derive_inner_full.take(&#take_ident)?
                    }
                    _ => return ::std::result::Result::Err(polars::prelude::polars_err!(
                        ComputeError: "df-derive: expected list AnyValue from __df_derive_vec_to_inner_list_values (codegen invariant violation)"
                    )),
                };
                __df_derive_out.push(polars::prelude::AnyValue::List(inner));
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
    let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
    let cols_buf_ident = syn::Ident::new("__df_derive_cols_buf", proc_macro2::Span::call_site());
    let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
    let per_item_vals_ident =
        syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());
    let recur_elem_ts = build_recur_elem(&elem_ident, &per_item_vals_ident);
    let wrap_extra_for_empty = gen_wrap_dtype_layers(vec_count(tail));

    quote! {{
        let #schema_ident = #ty::schema()?;
        let mut #cols_buf_ident: ::std::vec::Vec<::std::vec::Vec<polars::prelude::AnyValue>> =
            #schema_ident.iter().map(|_| ::std::vec::Vec::with_capacity((#acc).len())).collect();
        for #elem_ident in (#acc).iter() {
            let mut #per_item_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::new();
            { #recur_elem_ts }
            for (j, v) in #per_item_vals_ident.into_iter().enumerate() { #cols_buf_ident[j].push(v); }
        }
        let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(#schema_ident.len());
        for (j, (_inner_name, __df_derive_inner_dtype)) in #schema_ident.iter().enumerate() {
            let inner = if #cols_buf_ident[j].is_empty() {
                let mut __df_derive_wrapped = __df_derive_inner_dtype.clone();
                #wrap_extra_for_empty
                polars::prelude::Series::new_empty("".into(), &__df_derive_wrapped)
            } else {
                <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), &#cols_buf_ident[j])
            };
            __df_derive_out.push(polars::prelude::AnyValue::List(inner));
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
        generate_generic_for_anyvalue(ty, vals, &quote! { #elem }, tail)
    })
}

/// Trait-only on-leaf body for converting a single nested value into `AnyValues`
/// pushed onto `values_vec_ident` via `to_dataframe()`.
fn generic_leaf_to_anyvalues(values_vec_ident: &Ident, acc: &TokenStream) -> TokenStream {
    quote! {
        let __df_derive_tmp_df = (#acc).to_dataframe()?;
        let __df_derive_names: ::std::vec::Vec<String> = __df_derive_tmp_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for __df_derive_name in __df_derive_names.iter() {
            let __df_derive_v = __df_derive_tmp_df.column(__df_derive_name.as_str())?.get(0)?;
            #values_vec_ident.push(__df_derive_v.into_static());
        }
    }
}

/// Trait-only equivalent of `generate_nested_for_anyvalue` for generic params.
fn generate_generic_for_anyvalue(
    type_path: &TokenStream,
    values_vec_ident: &Ident,
    access: &TokenStream,
    wrappers: &[Wrapper],
) -> TokenStream {
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| generic_leaf_to_anyvalues(values_vec_ident, acc);

    let on_option_none = |_tail: &[Wrapper]| {
        quote! {
            let schema = #ty::schema()?;
            for _ in 0..schema.len() { #values_vec_ident.push(polars::prelude::AnyValue::Null); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let list_vals_ts = gen_generic_vec_to_list_anyvalues(&ty, acc, tail);
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for v in __df_derive_vals.into_iter() { #values_vec_ident.push(v); }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

// --- Context-specific generators ---

pub fn generate_nested_for_series(
    type_path: &TokenStream,
    series_name: &str,
    access: &TokenStream,
    wrappers: &[Wrapper],
    is_generic: bool,
) -> TokenStream {
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| {
        let main_logic = generate_scalar_struct_logic(series_name, acc);
        quote! { #main_logic }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let as_list_none = has_vec(tail);
        generate_null_series_for_struct(&ty, series_name, as_list_none)
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        if tail.is_empty() && !is_generic {
            quote! {{
                #ty::__df_derive_collect_vec_as_prefixed_list_series(&(#acc), #series_name)?
            }}
        } else {
            let schema_ident =
                syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
            let vals_ident = syn::Ident::new("__df_derive_vals", proc_macro2::Span::call_site());
            let list_vals_ts = if is_generic {
                gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
            } else {
                gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
            };
            quote! {{
                let #schema_ident = #ty::schema()?;
                let #vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
                let mut nested_series: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::with_capacity(#schema_ident.len());
                for (j, (inner_name, _dtype)) in #schema_ident.iter().enumerate() {
                    let prefixed_name = format!("{}.{}", #series_name, inner_name);
                    let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(prefixed_name.as_str().into(), &[#vals_ident[j].clone()]);
                    nested_series.push(s.into());
                }
                nested_series
            }}
        }
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_nested_for_columnar_push(
    type_path: &TokenStream,
    access: &TokenStream,
    wrappers: &[Wrapper],
    idx: usize,
    is_generic: bool,
) -> TokenStream {
    let ty = type_path.clone();
    let vec = has_vec(wrappers);

    // For non-vec shapes the populator is `Vec<Vec<AnyValue>>` (one inner
    // Vec per inner schema column, accumulating one AnyValue per outer row).
    // For vec shapes the populator is `Vec<Box<dyn ListBuilderTrait>>`
    // (one builder per inner schema column, accumulating one outer-list
    // entry per outer row). The on-leaf branch only runs for non-vec
    // shapes — `process_wrappers` reaches the leaf only when no `Vec`
    // wrapper is present.
    let cols_ident = PopulatorIdents::nested_struct_cols(idx);
    let lbs_ident = PopulatorIdents::nested_list_builders(idx);

    let on_leaf = |acc: &TokenStream| {
        let cols_ident = cols_ident.clone();
        if is_generic {
            quote! {
                let __df_derive_tmp_df = (#acc).to_dataframe()?;
                let __df_derive_names: ::std::vec::Vec<String> = __df_derive_tmp_df
                    .get_column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                for (j, __df_derive_name) in __df_derive_names.iter().enumerate() {
                    let __df_derive_v = __df_derive_tmp_df
                        .column(__df_derive_name.as_str())?
                        .get(0)?
                        .into_static();
                    #cols_ident[j].push(__df_derive_v);
                }
            }
        } else {
            quote! {
                let nested_values = (#acc).__df_derive_to_anyvalues()?;
                for (j, value) in nested_values.into_iter().enumerate() {
                    #cols_ident[j].push(value);
                }
            }
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        if vec {
            let lbs_ident = lbs_ident.clone();
            quote! {
                for j in 0..#lbs_ident.len() {
                    polars::prelude::ListBuilderTrait::append_null(&mut *#lbs_ident[j]);
                }
            }
        } else {
            let cols_ident = cols_ident.clone();
            quote! {
                for j in 0..#cols_ident.len() {
                    #cols_ident[j].push(polars::prelude::AnyValue::Null);
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
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for (j, __df_derive_v) in __df_derive_vals.into_iter().enumerate() {
                match __df_derive_v {
                    polars::prelude::AnyValue::List(__df_derive_inner) => {
                        polars::prelude::ListBuilderTrait::append_series(
                            &mut *#lbs_ident[j],
                            &__df_derive_inner,
                        )?;
                    }
                    polars::prelude::AnyValue::Null => {
                        polars::prelude::ListBuilderTrait::append_null(&mut *#lbs_ident[j]);
                    }
                    _ => {
                        return ::std::result::Result::Err(polars::prelude::polars_err!(
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
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| {
        quote! {
            let tmp_df = (#acc).to_dataframe()?;
            for col_name in tmp_df.get_column_names() {
                let v = tmp_df.column(col_name)?.get(0)?;
                #values_vec_ident.push(v.into_static());
            }
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        quote! {
            let schema = #ty::schema()?;
            for _ in 0..schema.len() { #values_vec_ident.push(polars::prelude::AnyValue::Null); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let list_vals_ts = if is_generic {
            gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
        } else {
            gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
        };
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
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
        let wrap_extra = gen_wrap_dtype_layers(vec_count(wrappers).saturating_sub(1));
        decls.push(quote! { let #schema_ident = #type_path::schema()?; });
        decls.push(quote! {
            let mut #lbs_ident: ::std::vec::Vec<
                ::std::boxed::Box<dyn polars::prelude::ListBuilderTrait>,
            > = #schema_ident
                .iter()
                .map(|(_, __df_derive_inner_dtype)| {
                    let mut __df_derive_wrapped = __df_derive_inner_dtype.clone();
                    #wrap_extra
                    polars::chunked_array::builder::get_list_builder(
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
            let mut #cols_ident: ::std::vec::Vec<::std::vec::Vec<polars::prelude::AnyValue>> =
                #schema_ident
                    .iter()
                    .map(|_| ::std::vec::Vec::with_capacity(items.len()))
                    .collect();
        });
    }
    decls
}

pub fn nested_finishers_for_vec_anyvalues(wrappers: &[Wrapper], idx: usize) -> TokenStream {
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
                let inner = polars::prelude::IntoSeries::into_series(
                    polars::prelude::ListBuilderTrait::finish(&mut *#lbs_ident[j]),
                );
                out_values.push(polars::prelude::AnyValue::List(inner));
            }
        }
    } else {
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        quote! {
            for j in 0..#schema_ident.len() {
                let inner = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), &#cols_ident[j]);
                out_values.push(polars::prelude::AnyValue::List(inner));
            }
        }
    }
}

pub fn nested_columnar_builders(
    wrappers: &[Wrapper],
    idx: usize,
    field_name: &str,
) -> Vec<TokenStream> {
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
                let full_name = format!("{}.{}", #name, col_name);
                let s = polars::prelude::IntoSeries::into_series(
                    polars::prelude::ListBuilderTrait::finish(&mut *#lbs_ident[j]),
                )
                .with_name(full_name.as_str().into());
                columns.push(s.into());
            }
        }}]
    } else {
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        vec![quote! {{
            for (j, (col_name, _)) in #schema_ident.iter().enumerate() {
                let full_name = format!("{}.{}", #name, col_name);
                let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(full_name.as_str().into(), &#cols_ident[j]);
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
    quote! {
        {
            let mut nested_fields: ::std::vec::Vec<(::std::string::String, polars::prelude::DataType)> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    polars::prelude::DataType::List(Box::new(inner_dtype))
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
    quote! {
        {
            let mut nested_series = Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    polars::prelude::DataType::List(Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                let empty_series = polars::prelude::Series::new_empty(prefixed_name.as_str().into(), &dtype);
                nested_series.push(empty_series.into());
            }
            nested_series
        }
    }
}

fn generate_null_series_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_series = Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    polars::prelude::DataType::List(Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                let null_series = polars::prelude::Series::new_empty(prefixed_name.as_str().into(), &dtype);
                let null_series_with_value = null_series.extend_constant(polars::prelude::AnyValue::Null, 1)?;
                nested_series.push(null_series_with_value.into());
            }
            nested_series
        }
    }
}

fn generate_scalar_struct_logic(column_name: &str, access_path: &TokenStream) -> TokenStream {
    quote! {
        {
            let nested_df = (#access_path).to_dataframe()?;
            let mut nested_series = Vec::new();

            for col_name in nested_df.get_column_names() {
                let prefixed_name = format!("{}.{}", #column_name, col_name);
                let series = nested_df.column(col_name)?.clone().with_name(prefixed_name.as_str().into());
                nested_series.push(series.into());
            }
            nested_series
        }
    }
}
