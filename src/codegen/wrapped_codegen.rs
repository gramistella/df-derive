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

/// True for `String` and `Option<String>` leaves with no transform — the cases
/// where the columnar populator buffer can borrow `&str` from `items` instead
/// of cloning each row's `String`. `Vec<_>` wrappers, transforms
/// (`ToString`, `DecimalToString`, …), and deeper nestings keep the existing
/// owning-buffer path.
fn is_borrowable_string_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    if !matches!(base, BaseType::String) || transform.is_some() {
        return false;
    }
    matches!(wrappers, [] | [Wrapper::Option])
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
    let base_is_struct = matches!(base_type, BaseType::Struct(..));

    // Recursively process a single vector element as a primitive and push one AnyValue
    let recur_elem_tokens = || {
        // Ensure we work with an owned primitive value regardless of reference
        let elem_owned_access = quote! { (*#elem_ident).clone() };
        super::wrapped_codegen::generate_primitive_for_anyvalue(
            &per_item_vals_ident,
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
    ty: &TokenStream,
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
                &per_item_vals_ident,
                &elem_access,
                tail,
                false,
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

/// Trait-only equivalent of `gen_nested_vec_to_list_anyvalues` for fields whose
/// base type is a generic type parameter. Avoids any inherent helpers and uses
/// only `ToDataFrame` / `Columnar` trait methods.
fn gen_generic_vec_to_list_anyvalues(
    ty: &TokenStream,
    acc: &TokenStream,
    tail: &[Wrapper],
) -> TokenStream {
    let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
    let cols_buf_ident = syn::Ident::new("__df_derive_cols_buf", proc_macro2::Span::call_site());
    let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
    let per_item_vals_ident =
        syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());

    let recur_elem_ts =
        generate_generic_for_anyvalue(ty, &per_item_vals_ident, &quote! { #elem_ident }, tail);

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
        for j in 0..#schema_ident.len() {
            let inner = polars::prelude::Series::new("".into(), &#cols_buf_ident[j]);
            __df_derive_out.push(polars::prelude::AnyValue::List(inner));
        }
        __df_derive_out
    }}
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

// --- Generic-field bulk emitters ---
//
// These produce the entire builder/finisher token stream for a generic-typed
// field (i.e., the field's base type is a type parameter of the enclosing
// struct). They sidestep the per-row push pipeline entirely by collecting
// values into a contiguous slice once and calling
// `<T as Columnar>::columnar_to_dataframe(slice)` exactly once. From there
// each schema column of `T` is sliced/scattered into the parent's columns or
// `AnyValue::List` entries.
//
// Three wrapper shapes are supported as bulk: the bare leaf (`payload: T`),
// `Option<T>`, and `Vec<T>`. Deeper nestings (`Option<Vec<T>>`, etc.) keep
// using the per-row trait-only paths defined elsewhere in this module — those
// are rare and the bulk variants would need offset+position bookkeeping that
// isn't worth the added complexity.

/// Codegen context for the bulk emitters. Each variant carries exactly the
/// data its consumer needs: the columnar arm prefixes inner Series names with
/// the parent field name and pushes them onto the `columns` Vec; the
/// vec-anyvalues arm wraps each inner Series in `AnyValue::List` and pushes
/// onto `out_values`. Keeping `parent_name` inside `Columnar` (rather than as
/// a separate parameter on `bulk_consume_inner_series`) prevents the vec
/// path from accidentally depending on a value it can't use.
#[derive(Clone, Copy)]
pub enum BulkContext<'a> {
    /// Builder-position emit inside `columnar_to_dataframe`. Pushes prefixed
    /// columns onto the in-scope `columns` Vec.
    Columnar { parent_name: &'a str },
    /// Finisher-position emit inside `__df_derive_vec_to_inner_list_values`.
    /// Pushes `AnyValue::List(inner)` onto the in-scope `out_values` Vec.
    VecAnyvalues,
}

/// Build the per-column emit body that adapts an inner `Series` to the given
/// context. `series_expr` must evaluate to an owned `polars::prelude::Series`.
///
/// Note: this helper assumes `<T as Columnar>::columnar_to_dataframe` returns a
/// `DataFrame` whose columns appear in the same order as `T::schema()` —
/// every call iterates `T::schema()` and looks up by column name, so a
/// mismatch wouldn't crash but would produce a parent `DataFrame` whose column
/// order silently diverges from the declared schema. The derive enforces this
/// for its own generated impls; user-written `Columnar` impls must do the
/// same to be compatible with derives that use the type as a generic
/// parameter.
fn bulk_consume_inner_series(
    ctx: BulkContext<'_>,
    col_name_var: &TokenStream,
    series_expr: &TokenStream,
) -> TokenStream {
    match ctx {
        BulkContext::Columnar { parent_name } => quote! {{
            let __df_derive_prefixed = format!("{}.{}", #parent_name, #col_name_var);
            let __df_derive_inner = #series_expr;
            let __df_derive_named = __df_derive_inner.with_name(__df_derive_prefixed.as_str().into());
            columns.push(__df_derive_named.into());
        }},
        BulkContext::VecAnyvalues => quote! {{
            let __df_derive_inner = #series_expr;
            out_values.push(polars::prelude::AnyValue::List(__df_derive_inner));
        }},
    }
}

/// Bulk emit for a leaf generic field (`payload: T`). Collects `Vec<T>` once,
/// calls `T::columnar_to_dataframe`, then ships each schema column to `ctx`.
pub fn gen_bulk_generic_leaf(
    ty: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let slice_ident = format_ident!("__df_derive_gen_slice_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let consume = bulk_consume_inner_series(
        ctx,
        &quote! { __df_derive_col_name },
        &quote! {
            #df_ident
                .column(__df_derive_col_name)?
                .as_materialized_series()
                .clone()
        },
    );
    quote! {{
        let #slice_ident: ::std::vec::Vec<#ty> = items
            .iter()
            .map(|__df_derive_it| (#access).clone())
            .collect();
        let #df_ident = #ty::columnar_to_dataframe(&#slice_ident)?;
        for (__df_derive_col_name, _) in #ty::schema()? {
            #consume
        }
    }}
}

/// Bulk emit for `payload: Option<T>`. Builds the gather (`nn`) plus position
/// (`pos`) vectors, calls `T::columnar_to_dataframe(&nn)` once (skipping when
/// every item is `None`), then scatters each `T` column back over the
/// original positions, emitting `AnyValue::Null` where the source was `None`.
pub fn gen_bulk_generic_option(
    ty: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let nn_ident = format_ident!("__df_derive_gen_nn_{}", idx);
    let pos_ident = format_ident!("__df_derive_gen_pos_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };

    let consume_filled = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_inner_col = #df_ident.column(#col_name_var)?;
            let mut __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> =
                ::std::vec::Vec::with_capacity(items.len());
            for __df_derive_ix in #pos_ident.iter() {
                match __df_derive_ix {
                    ::std::option::Option::Some(k) => {
                        __df_derive_vals.push(__df_derive_inner_col.get(*k)?.into_static());
                    }
                    ::std::option::Option::None => {
                        __df_derive_vals.push(polars::prelude::AnyValue::Null);
                    }
                }
            }
            polars::prelude::Series::new("".into(), &__df_derive_vals)
        }},
    );
    let consume_empty = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {
            polars::prelude::Series::new_empty("".into(), &#dtype_var)
                .extend_constant(polars::prelude::AnyValue::Null, items.len())?
        },
    );

    quote! {{
        let mut #nn_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<usize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for __df_derive_it in items {
            match &(#access) {
                ::std::option::Option::Some(__df_derive_v) => {
                    #pos_ident.push(::std::option::Option::Some(#nn_ident.len()));
                    #nn_ident.push(__df_derive_v.clone());
                }
                ::std::option::Option::None => {
                    #pos_ident.push(::std::option::Option::None);
                }
            }
        }
        if #nn_ident.is_empty() {
            for (#col_name_var, #dtype_var) in #ty::schema()? {
                #consume_empty
            }
        } else {
            let #df_ident = #ty::columnar_to_dataframe(&#nn_ident)?;
            for (#col_name_var, _) in #ty::schema()? {
                #consume_filled
            }
        }
    }}
}

/// Bulk emit for `payload: Vec<T>`. Flattens every parent row's `Vec<T>` into
/// a single contiguous slice plus a `lengths`-style offsets array, calls
/// `T::columnar_to_dataframe(&flat)` once, then slices each `T` column per
/// parent row to build a list-of-lists series (or list-of-list `AnyValue` for
/// the vec-anyvalues path).
pub fn gen_bulk_generic_vec(
    ty: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };

    let consume_filled = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_inner_col = #df_ident
                .column(#col_name_var)?
                .as_materialized_series()
                .clone();
            let mut __df_derive_rows: ::std::vec::Vec<polars::prelude::AnyValue> =
                ::std::vec::Vec::with_capacity(items.len());
            for __df_derive_i in 0..items.len() {
                let __df_derive_start = #offsets_ident[__df_derive_i];
                let __df_derive_end = #offsets_ident[__df_derive_i + 1];
                let __df_derive_slice = __df_derive_inner_col
                    .slice(__df_derive_start as i64, __df_derive_end - __df_derive_start);
                __df_derive_rows.push(polars::prelude::AnyValue::List(__df_derive_slice));
            }
            polars::prelude::Series::new("".into(), &__df_derive_rows)
        }},
    );
    let consume_empty = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let mut __df_derive_rows: ::std::vec::Vec<polars::prelude::AnyValue> =
                ::std::vec::Vec::with_capacity(items.len());
            for _ in 0..items.len() {
                let __df_derive_empty = polars::prelude::Series::new_empty("".into(), &#dtype_var);
                __df_derive_rows.push(polars::prelude::AnyValue::List(__df_derive_empty));
            }
            polars::prelude::Series::new("".into(), &__df_derive_rows)
        }},
    );

    quote! {{
        let mut #flat_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        let mut #offsets_ident: ::std::vec::Vec<usize> = ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_v in (&(#access)).iter() {
                #flat_ident.push((*__df_derive_v).clone());
            }
            #offsets_ident.push(#flat_ident.len());
        }
        if #flat_ident.is_empty() {
            for (#col_name_var, #dtype_var) in #ty::schema()? {
                #consume_empty
            }
        } else {
            let #df_ident = #ty::columnar_to_dataframe(&#flat_ident)?;
            for (#col_name_var, _) in #ty::schema()? {
                #consume_filled
            }
        }
    }}
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
        // Borrowing fast path for `String` leaves with no transform: build the
        // 1-row Series from `&[&str]` so the per-row `to_dataframe(&self)` API
        // doesn't clone the field's `String` before handing it to Polars.
        if matches!(base_type, BaseType::String) && transform.is_none() {
            return quote! {
                vec![{
                    let s = polars::prelude::Series::new(
                        #series_name.into(),
                        &[(#acc).as_str()],
                    );
                    s.into()
                }]
            };
        }
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
    // Borrowing fast path for `String` / `Option<String>`: the buffer is
    // declared as `Vec<&str>` / `Vec<Option<&str>>` by `primitive_decls`, so we
    // push borrows of the field instead of cloning each row's `String` into an
    // owned buffer. The borrows live as long as `items`, which outlives the
    // buffer.
    if is_borrowable_string_leaf(base_type, transform, wrappers) {
        let vec_ident = format_ident!("__df_derive_buf_{}", idx);
        return if is_option(wrappers) {
            quote! { #vec_ident.push((#access).as_deref()); }
        } else {
            quote! { #vec_ident.push(&(#access)); }
        };
    }

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
    values_vec_ident: &Ident,
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
        // Borrowing fast path for `String` leaves with no transform: skip the
        // user-side clone by building the 1-element Series from `&[&str]`.
        // The Series owns its own Arrow buffer once `from_slice` returns, so
        // `s.get(0)?.into_static()` is safe to call after the borrow's scope.
        if matches!(base_type, BaseType::String) && transform.is_none() {
            return quote! {
                let s = polars::prelude::Series::new(
                    "".into(),
                    &[(#acc).as_str()],
                );
                #values_vec_ident.push(s.get(0)?.into_static());
            };
        }
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
    type_path: &TokenStream,
    series_name: &str,
    access: &TokenStream,
    wrappers: &[Wrapper],
    is_generic: bool,
) -> TokenStream {
    #![allow(clippy::too_many_lines)]
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| {
        let main_logic = generate_scalar_struct_logic(series_name, acc);
        quote! { #main_logic }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let as_list_none = tail.iter().any(|w| matches!(w, Wrapper::Vec));
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
    type_path: &TokenStream,
    access: &TokenStream,
    wrappers: &[Wrapper],
    idx: usize,
    is_generic: bool,
) -> TokenStream {
    let ty = type_path.clone();

    let cols_ident = if is_vec(wrappers) {
        format_ident!("__df_derive_nv_cols_{}", idx)
    } else {
        format_ident!("__df_derive_ns_cols_{}", idx)
    };

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
        let cols_ident = cols_ident.clone();
        quote! { for j in 0..#cols_ident.len() { #cols_ident[j].push(polars::prelude::AnyValue::Null); } }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let cols_ident = cols_ident.clone();
        let list_vals_ts = if is_generic {
            gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
        } else {
            gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
        };
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for (j, v) in __df_derive_vals.into_iter().enumerate() { #cols_ident[j].push(v); }
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

    // Borrowing fast path for `String` / `Option<String>`: a `Vec<&str>` (or
    // `Vec<Option<&str>>`) buffer borrows from `items` instead of cloning each
    // row's `String`. `Series::new(name, &Vec<&str>)` dispatches to
    // `StringChunked::from_slice` and produces the same `Utf8ViewArray`-backed
    // column the owning path produces.
    if is_borrowable_string_leaf(base_type, transform, wrappers) {
        let vec_ident = format_ident!("__df_derive_buf_{}", idx);
        if opt {
            decls.push(quote! {
                let mut #vec_ident: ::std::vec::Vec<::std::option::Option<&str>> =
                    ::std::vec::Vec::with_capacity(items.len());
            });
        } else {
            decls.push(quote! {
                let mut #vec_ident: ::std::vec::Vec<&str> =
                    ::std::vec::Vec::with_capacity(items.len());
            });
        }
        return decls;
    }

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
    type_path: &TokenStream,
    name: &str,
    wrappers: &[Wrapper],
) -> TokenStream {
    generate_empty_series_for_struct(type_path, name, is_vec(wrappers))
}

pub fn nested_decls(wrappers: &[Wrapper], type_path: &TokenStream, idx: usize) -> Vec<TokenStream> {
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
    decls.push(quote! { let #schema_ident = #type_path::schema()?; });
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
    type_path: &TokenStream,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_fields: Vec<(&'static str, polars::prelude::DataType)> = Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
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
