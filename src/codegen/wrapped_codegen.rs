use crate::ir::{BaseType, PrimitiveTransform, Wrapper};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

fn is_vec(wrappers: &[Wrapper]) -> bool {
    wrappers.iter().any(|w| matches!(w, Wrapper::Vec))
}

fn is_option(wrappers: &[Wrapper]) -> bool {
    wrappers.iter().any(|w| matches!(w, Wrapper::Option))
}

// --- Context-specific generation APIs ---

// --- Helpers to unify Vec processing across targets ---

/// Build tokens that evaluate to a `polars::prelude::Series` representing
/// the inner list for a primitive vector (including nested wrappers in `tail`).
fn gen_primitive_vec_inner_series(
    acc: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    tail: &[Wrapper],
) -> TokenStream {
    let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
    let per_item_vals_ident =
        syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());
    let list_vals_ident = syn::Ident::new("__df_derive_list_vals", proc_macro2::Span::call_site());

    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, tail);
    let elem_dtype = mapping.element_dtype;
    let do_cast = crate::codegen::type_registry::needs_cast(transform);
    let fast_inner_ts = super::common::generate_inner_series_from_vec(acc, base_type, transform);
    let base_is_struct = matches!(base_type, BaseType::Struct(_));

    // Recursively process a single vector element as a primitive and push one AnyValue
    let recur_elem_tokens = || {
        // Ensure we work with an owned primitive value regardless of reference
        let elem_owned_access = quote! { (*#elem_ident).clone() };
        super::wrapped_codegen::generate_primitive_for_anyvalue(
            per_item_vals_ident.clone(),
            &elem_owned_access,
            base_type,
            transform,
            tail,
        )
    };
    let recur_elem_tokens_ts = recur_elem_tokens();

    if !base_is_struct && tail.is_empty() {
        quote! {{ { #fast_inner_ts } }}
    } else if !base_is_struct
        && tail.len() == 1
        && matches!(tail[0], Wrapper::Option)
        && transform.is_none()
    {
        quote! {{ polars::prelude::Series::new("".into(), &(#acc)) }}
    } else {
        quote! {{
            let mut #list_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity((#acc).len());
            for #elem_ident in (#acc).iter() {
                let mut #per_item_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::new();
                { #recur_elem_tokens_ts }
                #list_vals_ident.push(#per_item_vals_ident.pop().expect("expected single AnyValue for primitive element"));
            }
            let mut __df_derive_inner = polars::prelude::Series::new("".into(), &#list_vals_ident);
            if #do_cast { __df_derive_inner = __df_derive_inner.cast(&#elem_dtype)?; }
            __df_derive_inner
        }}
    }
}

/// Build tokens that evaluate to `Vec<polars::prelude::AnyValue>`, where each element
/// is a `AnyValue::List(inner_series)` for one nested struct field across the vector.
fn gen_nested_vec_to_list_anyvalues(
    ty: &Ident,
    acc: &TokenStream,
    tail: &[Wrapper],
) -> TokenStream {
    if tail.is_empty() {
        // Fast path: Vec<Struct>
        quote! { #ty::__df_derive_vec_to_inner_list_values(&(#acc))? }
    } else if tail.len() == 1 && matches!(tail[0], Wrapper::Option) {
        // Semi-optimized: Vec<Option<Struct>>
        let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
        let pos_ident = syn::Ident::new("__df_derive_pos", proc_macro2::Span::call_site());
        let nn_ident = syn::Ident::new("__df_derive_nn", proc_macro2::Span::call_site());
        let vals_ident = syn::Ident::new("__df_derive_vals", proc_macro2::Span::call_site());
        quote! {{
            let #schema_ident = #ty::schema()?;
            let mut #pos_ident: ::std::vec::Vec<::std::option::Option<usize>> = ::std::vec::Vec::with_capacity((#acc).len());
            let mut #nn_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
            for __df_derive_maybe in (#acc).iter() {
                match __df_derive_maybe {
                    ::std::option::Option::Some(v) => {
                        #pos_ident.push(::std::option::Option::Some(#nn_ident.len()));
                        #nn_ident.push((*v).clone());
                    }
                    ::std::option::Option::None => #pos_ident.push(::std::option::Option::None),
                }
            }
            if #nn_ident.is_empty() {
                let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(#schema_ident.len());
                for _ in 0..#schema_ident.len() {
                    let __df_derive_nulls: ::std::vec::Vec<polars::prelude::AnyValue> = (0..(#acc).len()).map(|_| polars::prelude::AnyValue::Null).collect();
                    let inner = polars::prelude::Series::new("".into(), &__df_derive_nulls);
                    __df_derive_out.push(polars::prelude::AnyValue::List(inner));
                }
                __df_derive_out
            } else {
                let #vals_ident = #ty::__df_derive_vec_to_inner_list_values(&#nn_ident)?;
                let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(#schema_ident.len());
                for j in 0..#schema_ident.len() {
                    let __df_derive_list_vals: ::std::vec::Vec<polars::prelude::AnyValue> = match &#vals_ident[j] {
                        polars::prelude::AnyValue::List(inner) => {
                            let mut out: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(#pos_ident.len());
                            for __df_derive_ix in #pos_ident.iter() {
                                if let ::std::option::Option::Some(k) = __df_derive_ix {
                                    let v = inner.get(*k)?;
                                    out.push(v.into_static());
                                } else {
                                    out.push(polars::prelude::AnyValue::Null);
                                }
                            }
                            out
                        }
                        _ => unreachable!("expected list AnyValue for vec_to_inner_list_values"),
                    };
                    let inner = polars::prelude::Series::new("".into(), &__df_derive_list_vals);
                    __df_derive_out.push(polars::prelude::AnyValue::List(inner));
                }
                __df_derive_out
            }
        }}
    } else {
        // Fallback: recursive per-element
        let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
        let cols_buf_ident =
            syn::Ident::new("__df_derive_cols_buf", proc_macro2::Span::call_site());
        let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
        let per_item_vals_ident =
            syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());

        let recur_elem = || {
            let elem_access = quote! { #elem_ident };
            super::wrapped_codegen::generate_nested_for_anyvalue(
                ty,
                per_item_vals_ident.clone(),
                &elem_access,
                tail,
            )
        };
        let recur_elem_ts = recur_elem();

        quote! {{
            let #schema_ident = #ty::schema()?;
            let mut #cols_buf_ident: ::std::vec::Vec<::std::vec::Vec<polars::prelude::AnyValue>> = #schema_ident.iter().map(|_| ::std::vec::Vec::with_capacity((#acc).len())).collect();
            for #elem_ident in (#acc).iter() {
                let mut #per_item_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::new();
                { #recur_elem_ts }
                for (j, v) in #per_item_vals_ident.into_iter().enumerate() { #cols_buf_ident[j].push(v); }
            }
            let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(#schema_ident.len());
            for j in 0..#schema_ident.len() {
                let inner = polars::prelude::Series::new("".into(), &#cols_buf_ident[j]);
                __df_derive_out.push(polars::prelude::AnyValue::List(inner));
            }
            __df_derive_out
        }}
    }
}

// --- Primitive: context-specific generators ---

#[allow(clippy::too_many_lines)]
pub fn generate_primitive_for_series(
    series_name: &str,
    access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TokenStream {
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let dtype = mapping.full_dtype.clone();
    let elem_rust_ty = mapping.rust_element_type;
    let do_cast = crate::codegen::type_registry::needs_cast(transform);

    let on_leaf = |acc: &TokenStream| {
        let mapped = super::common::generate_primitive_access_expr(acc, transform);
        let cast_ts = if do_cast {
            quote! { s = s.cast(&#dtype)?; }
        } else {
            quote! {}
        };
        quote! {
            vec![{
                let mut s = polars::prelude::Series::new(#series_name.into(), ::std::slice::from_ref(&{ #mapped }));
                #cast_ts
                s.into()
            }]
        }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let tail_has_vec = tail.iter().any(|w| matches!(w, Wrapper::Vec));
        if tail_has_vec {
            quote! {
                vec![{
                    let list_any_value = polars::prelude::AnyValue::Null;
                    polars::prelude::Series::new(#series_name.into(), &[list_any_value]).into()
                }]
            }
        } else {
            quote! {
                vec![{
                    let __df_derive_tmp_opt: ::std::option::Option<#elem_rust_ty> = ::std::option::Option::None;
                    let mut s = polars::prelude::Series::new(#series_name.into(), std::slice::from_ref(&__df_derive_tmp_opt));
                    if #do_cast { s = s.cast(&#dtype)?; }
                    s.into()
                }]
            }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        quote! {{
            let inner_series = { #inner_series_ts };
            vec![{
                let list_any_value = polars::prelude::AnyValue::List(inner_series);
                polars::prelude::Series::new(#series_name.into(), &[list_any_value]).into()
            }]
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_primitive_for_columnar_push(
    access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    idx: usize,
) -> TokenStream {
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let _dtype = mapping.full_dtype.clone();
    let _elem_rust_ty = mapping.rust_element_type;
    let _do_cast = crate::codegen::type_registry::needs_cast(transform);
    let opt_scalar = is_option(wrappers) && !is_vec(wrappers);

    let on_leaf = |acc: &TokenStream| {
        let vec_ident = format_ident!("__df_derive_buf_{}", idx);
        let mapped = super::common::generate_primitive_access_expr(acc, transform);
        if opt_scalar {
            quote! { #vec_ident.push(::std::option::Option::Some({ #mapped })); }
        } else {
            quote! { #vec_ident.push({ #mapped }); }
        }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let tail_has_vec = tail.iter().any(|w| matches!(w, Wrapper::Vec));
        if tail_has_vec {
            let vals_ident = format_ident!("__df_derive_pv_vals_{}", idx);
            quote! { #vals_ident.push(polars::prelude::AnyValue::Null); }
        } else {
            let vec_ident = format_ident!("__df_derive_buf_{}", idx);
            quote! { #vec_ident.push(::std::option::Option::None); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        let vals_ident = format_ident!("__df_derive_pv_vals_{}", idx);
        quote! {{
            let inner_series = { #inner_series_ts };
            #vals_ident.push(polars::prelude::AnyValue::List(inner_series));
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_primitive_for_anyvalue(
    values_vec_ident: Ident,
    access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TokenStream {
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let dtype = mapping.full_dtype.clone();
    let _elem_rust_ty = mapping.rust_element_type;
    let do_cast = crate::codegen::type_registry::needs_cast(transform);

    let on_leaf = |acc: &TokenStream| {
        let mapped = super::common::generate_primitive_access_expr(acc, transform);
        let cast_ts = if do_cast {
            quote! { s = s.cast(&#dtype)?; }
        } else {
            quote! {}
        };
        quote! {
            let mut s = polars::prelude::Series::new("".into(), std::slice::from_ref(&{ #mapped }));
            #cast_ts
            #values_vec_ident.push(s.get(0)?.into_static());
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        quote! { #values_vec_ident.push(polars::prelude::AnyValue::Null); }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        quote! {{
            let inner_series = { #inner_series_ts };
            #values_vec_ident.push(polars::prelude::AnyValue::List(inner_series));
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

// --- Nested: context-specific generators ---

pub fn generate_nested_for_series(
    type_ident: &Ident,
    series_name: &str,
    access: &TokenStream,
    wrappers: &[Wrapper],
) -> TokenStream {
    #![allow(clippy::too_many_lines)]
    let ty = type_ident.clone();

    let on_leaf = |acc: &TokenStream| {
        let main_logic = generate_scalar_struct_logic(series_name, acc);
        quote! { #main_logic }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let as_list_none = tail.iter().any(|w| matches!(w, Wrapper::Vec));
        generate_null_series_for_struct(&ty, series_name, as_list_none)
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        if tail.is_empty() {
            quote! {{
                #ty::__df_derive_collect_vec_as_prefixed_list_series(&(#acc), #series_name)?
            }}
        } else {
            let schema_ident =
                syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
            let vals_ident = syn::Ident::new("__df_derive_vals", proc_macro2::Span::call_site());
            let list_vals_ts = gen_nested_vec_to_list_anyvalues(&ty, acc, tail);
            quote! {{
                let #schema_ident = #ty::schema()?;
                let #vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
                let mut nested_series: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::with_capacity(#schema_ident.len());
                for (j, (inner_name, _dtype)) in #schema_ident.iter().enumerate() {
                    let prefixed_name = format!("{}.{}", #series_name, inner_name);
                    let s = polars::prelude::Series::new(prefixed_name.as_str().into(), &[#vals_ident[j].clone()]);
                    nested_series.push(s.into());
                }
                nested_series
            }}
        }
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_nested_for_columnar_push(
    type_ident: &Ident,
    access: &TokenStream,
    wrappers: &[Wrapper],
    idx: usize,
) -> TokenStream {
    let ty = type_ident.clone();

    let on_leaf = |acc: &TokenStream| {
        let cols_ident = if is_vec(wrappers) {
            format_ident!("__df_derive_nv_cols_{}", idx)
        } else {
            format_ident!("__df_derive_ns_cols_{}", idx)
        };
        quote! {
            let nested_values = (#acc).__df_derive_to_anyvalues()?;
            for (j, value) in nested_values.into_iter().enumerate() {
                #cols_ident[j].push(value);
            }
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        let cols_ident = if is_vec(wrappers) {
            format_ident!("__df_derive_nv_cols_{}", idx)
        } else {
            format_ident!("__df_derive_ns_cols_{}", idx)
        };
        quote! { for j in 0..#cols_ident.len() { #cols_ident[j].push(polars::prelude::AnyValue::Null); } }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let cols_ident = if is_vec(wrappers) {
            format_ident!("__df_derive_nv_cols_{}", idx)
        } else {
            format_ident!("__df_derive_ns_cols_{}", idx)
        };
        let list_vals_ts = gen_nested_vec_to_list_anyvalues(&ty, acc, tail);
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for (j, v) in __df_derive_vals.into_iter().enumerate() { #cols_ident[j].push(v); }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_nested_for_anyvalue(
    type_ident: &Ident,
    values_vec_ident: Ident,
    access: &TokenStream,
    wrappers: &[Wrapper],
) -> TokenStream {
    let ty = type_ident.clone();

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
        let list_vals_ts = gen_nested_vec_to_list_anyvalues(&ty, acc, tail);
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for v in __df_derive_vals.into_iter() { #values_vec_ident.push(v); }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}
// --- Primitive: Columnar (centralized move from wrapper_logic.rs) ---

pub fn primitive_decls(
    wrappers: &[Wrapper],
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    idx: usize,
) -> Vec<TokenStream> {
    let mut decls: Vec<TokenStream> = Vec::new();
    let opt = is_option(wrappers);
    let vec = is_vec(wrappers);
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let elem_rust_ty = mapping.rust_element_type;
    if vec {
        let vals_ident = format_ident!("__df_derive_pv_vals_{}", idx);
        decls.push(quote! { let mut #vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(items.len()); });
    } else {
        let vec_ident = format_ident!("__df_derive_buf_{}", idx);
        if opt {
            decls.push(quote! { let mut #vec_ident: ::std::vec::Vec<::std::option::Option<#elem_rust_ty>> = ::std::vec::Vec::with_capacity(items.len()); });
        } else {
            decls.push(quote! { let mut #vec_ident: ::std::vec::Vec<#elem_rust_ty> = ::std::vec::Vec::with_capacity(items.len()); });
        }
    }
    decls
}

pub fn primitive_finishers_for_vec_anyvalues(
    wrappers: &[Wrapper],
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    idx: usize,
) -> TokenStream {
    let vec = is_vec(wrappers);
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let dtype = if vec {
        mapping.element_dtype
    } else {
        mapping.full_dtype
    };
    let needs_cast = crate::codegen::type_registry::needs_cast(transform);
    if vec {
        let vals_ident = format_ident!("__df_derive_pv_vals_{}", idx);
        quote! {
            let inner = polars::prelude::Series::new("".into(), &#vals_ident);
            out_values.push(polars::prelude::AnyValue::List(inner));
        }
    } else {
        let vec_ident = format_ident!("__df_derive_buf_{}", idx);
        quote! {
            let mut inner = polars::prelude::Series::new("".into(), &#vec_ident);
            if #needs_cast { inner = inner.cast(&#dtype)?; }
            out_values.push(polars::prelude::AnyValue::List(inner));
        }
    }
}

// --- Nested Structs: Row-wise and Columnar (centralized) ---

pub fn nested_empty_series_row(
    type_ident: &Ident,
    name: &str,
    wrappers: &[Wrapper],
) -> TokenStream {
    generate_empty_series_for_struct(type_ident, name, is_vec(wrappers))
}

pub fn nested_decls(wrappers: &[Wrapper], type_ident: &Ident, idx: usize) -> Vec<TokenStream> {
    let mut decls: Vec<TokenStream> = Vec::new();
    let vec = is_vec(wrappers);
    let schema_ident = if vec {
        format_ident!("__df_derive_nv_schema_{}", idx)
    } else {
        format_ident!("__df_derive_ns_schema_{}", idx)
    };
    let cols_ident = if vec {
        format_ident!("__df_derive_nv_cols_{}", idx)
    } else {
        format_ident!("__df_derive_ns_cols_{}", idx)
    };
    decls.push(quote! { let #schema_ident = #type_ident::schema()?; });
    decls.push(quote! { let mut #cols_ident: ::std::vec::Vec<::std::vec::Vec<polars::prelude::AnyValue>> = #schema_ident.iter().map(|_| ::std::vec::Vec::with_capacity(items.len())).collect(); });
    decls
}

pub fn nested_finishers_for_vec_anyvalues(wrappers: &[Wrapper], idx: usize) -> TokenStream {
    let vec = is_vec(wrappers);
    let schema_ident = if vec {
        format_ident!("__df_derive_nv_schema_{}", idx)
    } else {
        format_ident!("__df_derive_ns_schema_{}", idx)
    };
    let cols_ident = if vec {
        format_ident!("__df_derive_nv_cols_{}", idx)
    } else {
        format_ident!("__df_derive_ns_cols_{}", idx)
    };
    quote! {
        for j in 0..#schema_ident.len() {
            let inner = polars::prelude::Series::new("".into(), &#cols_ident[j]);
            out_values.push(polars::prelude::AnyValue::List(inner));
        }
    }
}

pub fn nested_columnar_builders(
    wrappers: &[Wrapper],
    idx: usize,
    field_name: &str,
) -> Vec<TokenStream> {
    let vec = is_vec(wrappers);
    let schema_ident = if vec {
        format_ident!("__df_derive_nv_schema_{}", idx)
    } else {
        format_ident!("__df_derive_ns_schema_{}", idx)
    };
    let cols_ident = if vec {
        format_ident!("__df_derive_nv_cols_{}", idx)
    } else {
        format_ident!("__df_derive_ns_cols_{}", idx)
    };
    let name = field_name;
    vec![quote! {{
        for (j, (col_name, _)) in #schema_ident.iter().enumerate() {
            let full_name = format!("{}.{}", #name, col_name);
            let s = polars::prelude::Series::new(full_name.as_str().into(), &#cols_ident[j]);
            columns.push(s.into());
        }
    }}]
}

pub fn generate_schema_entries_for_struct(
    type_ident: &Ident,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_fields: Vec<(&'static str, polars::prelude::DataType)> = Vec::new();
            for (inner_name, inner_dtype) in #type_ident::schema()? {
                let leaked_name: &'static str = {
                    let s = format!("{}.{}", #column_name, inner_name);
                    Box::leak(s.into_boxed_str())
                };
                let dtype = if #as_list {
                    polars::prelude::DataType::List(Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                nested_fields.push((leaked_name, dtype));
            }
            nested_fields
        }
    }
}

fn generate_empty_series_for_struct(
    type_ident: &Ident,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_series = Vec::new();
            for (inner_name, inner_dtype) in #type_ident::schema()? {
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
    type_ident: &Ident,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_series = Vec::new();
            for (inner_name, inner_dtype) in #type_ident::schema()? {
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
