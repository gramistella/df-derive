// Generic-field bulk emitters.
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
// using the per-row trait-only paths defined elsewhere; those are rare and
// the bulk variants would need offset+position bookkeeping that isn't worth
// the added complexity.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

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
        BulkContext::VecAnyvalues => {
            let pp = super::polars_paths::prelude();
            quote! {{
                let __df_derive_inner = #series_expr;
                out_values.push(#pp::AnyValue::List(__df_derive_inner));
            }}
        }
    }
}

/// Bulk emit for a leaf generic field (`payload: T`). Collects `Vec<T>` once,
/// calls `T::columnar_to_dataframe`, then ships each schema column to `ctx`.
pub fn gen_bulk_generic_leaf(
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
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
        let #df_ident = <#ty as #columnar_trait>::columnar_to_dataframe(&#slice_ident)?;
        for (__df_derive_col_name, _) in <#ty as #to_df_trait>::schema()? {
            let __df_derive_col_name: &str = __df_derive_col_name.as_str();
            #consume
        }
    }}
}

/// Bulk emit for `payload: Option<T>`. Builds the gather (`nn`) plus position
/// (`pos`) vectors, calls `T::columnar_to_dataframe(&nn)` once (skipping when
/// every item is `None`), then scatters each `T` column back over the
/// original positions, emitting nulls where the source was `None`. Builds the
/// scattered Series via `Series::take` over an indices `IdxCa` instead of a
/// `Vec<AnyValue>` round-trip — typed buffers stay typed and we skip the
/// `AnyValue` dispatch per element.
pub fn gen_bulk_generic_option(
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let nn_ident = format_ident!("__df_derive_gen_nn_{}", idx);
    let pos_ident = format_ident!("__df_derive_gen_pos_{}", idx);
    let take_ident = format_ident!("__df_derive_gen_take_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };

    let consume_filled = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_inner_col = #df_ident
                .column(#col_name_var)?
                .as_materialized_series();
            __df_derive_inner_col.take(&#take_ident)?
        }},
    );
    let consume_empty = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {
            #pp::Series::new_empty("".into(), &#dtype_var)
                .extend_constant(#pp::AnyValue::Null, items.len())?
        },
    );

    quote! {{
        let mut #nn_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for __df_derive_it in items {
            match &(#access) {
                ::std::option::Option::Some(__df_derive_v) => {
                    #pos_ident.push(::std::option::Option::Some(
                        #nn_ident.len() as #pp::IdxSize,
                    ));
                    #nn_ident.push(__df_derive_v.clone());
                }
                ::std::option::Option::None => {
                    #pos_ident.push(::std::option::Option::None);
                }
            }
        }
        if #nn_ident.is_empty() {
            for (#col_name_var, #dtype_var) in <#ty as #to_df_trait>::schema()? {
                let #col_name_var: &str = #col_name_var.as_str();
                let #dtype_var = &#dtype_var;
                #consume_empty
            }
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_to_dataframe(&#nn_ident)?;
            let #take_ident: #pp::IdxCa =
                <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #pos_ident.iter().copied(),
                );
            for (#col_name_var, _) in <#ty as #to_df_trait>::schema()? {
                let #col_name_var: &str = #col_name_var.as_str();
                #consume_filled
            }
        }
    }}
}

/// Build the per-inner-column emit for a bulk-vec context: given the inner
/// `DataFrame` (already computed from the flat slice) plus the `offsets`
/// array, slice the inner column per parent row and feed each slice to a
/// typed `ListBuilder`. The finisher produces a `ListChunked` that goes
/// straight into either the parent `columns` Vec or an outer `AnyValue::List`.
///
/// Going through `ListBuilderTrait` directly skips the dtype-inference scan
/// and per-row `cast(inner_type)` Polars does inside
/// `any_values_to_list` when consuming a `Vec<AnyValue::List>`. The
/// `ListPrimitiveChunkedBuilder` / `ListStringChunkedBuilder` selected by
/// `get_list_builder` for primitive inner dtypes copies elements with
/// `extend_from_slice` rather than per-element `AnyValue` dispatch; the
/// `AnonymousOwnedListBuilder` selected for struct/list inner dtypes
/// Arc-shares chunks rather than wrapping them in an `AnyValue` envelope.
fn bulk_vec_consume_inner_columns(
    ctx: BulkContext<'_>,
    df_ident: &Ident,
    offsets_ident: &Ident,
    schema_iter_ts: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let cab = super::polars_paths::chunked_array_builder();
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let consume_filled = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_inner_col = #df_ident
                .column(#col_name_var)?
                .as_materialized_series();
            let mut __df_derive_lb = #cab::get_list_builder(
                #dtype_var,
                __df_derive_inner_col.len(),
                items.len(),
                "".into(),
            );
            for __df_derive_i in 0..items.len() {
                let __df_derive_start = #offsets_ident[__df_derive_i];
                let __df_derive_end = #offsets_ident[__df_derive_i + 1];
                let __df_derive_slice = __df_derive_inner_col
                    .slice(__df_derive_start as i64, __df_derive_end - __df_derive_start);
                #pp::ListBuilderTrait::append_series(
                    &mut *__df_derive_lb,
                    &__df_derive_slice,
                )?;
            }
            #pp::IntoSeries::into_series(
                #pp::ListBuilderTrait::finish(&mut *__df_derive_lb),
            )
        }},
    );
    quote! {
        for (#col_name_var, #dtype_var) in #schema_iter_ts {
            let #col_name_var: &str = #col_name_var.as_str();
            let #dtype_var: &#pp::DataType = &#dtype_var;
            #consume_filled
        }
    }
}

/// Build the all-empty-rows emit for a bulk-vec context: when every parent
/// row's inner Vec is empty (so the flat slice is empty and we skip the
/// inner `columnar_to_dataframe` call), still produce one outer-list column
/// per inner schema entry, each containing `items.len()` empty inner lists.
fn bulk_vec_consume_empty_columns(
    ctx: BulkContext<'_>,
    schema_iter_ts: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let cab = super::polars_paths::chunked_array_builder();
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let consume_empty = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_empty = #pp::Series::new_empty("".into(), #dtype_var);
            let mut __df_derive_lb = #cab::get_list_builder(
                #dtype_var,
                0,
                items.len(),
                "".into(),
            );
            for _ in 0..items.len() {
                #pp::ListBuilderTrait::append_series(
                    &mut *__df_derive_lb,
                    &__df_derive_empty,
                )?;
            }
            #pp::IntoSeries::into_series(
                #pp::ListBuilderTrait::finish(&mut *__df_derive_lb),
            )
        }},
    );
    quote! {
        for (#col_name_var, #dtype_var) in #schema_iter_ts {
            let #col_name_var: &str = #col_name_var.as_str();
            let #dtype_var: &#pp::DataType = &#dtype_var;
            #consume_empty
        }
    }
}

/// Bulk emit for `payload: Vec<T>` where `T` is a generic type parameter.
/// Flattens via `T: Clone` (already a macro-injected bound on `T`), calls
/// `<T as Columnar>::columnar_to_dataframe(&flat)` once, then ships each
/// inner column to `ctx` via `bulk_vec_consume_inner_columns`.
pub fn gen_bulk_generic_vec(
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let consume_filled =
        bulk_vec_consume_inner_columns(ctx, &df_ident, &offsets_ident, &schema_iter);
    let consume_empty = bulk_vec_consume_empty_columns(ctx, &schema_iter);

    quote! {{
        let mut #flat_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        let mut #offsets_ident: ::std::vec::Vec<usize> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_v in (&(#access)).iter() {
                #flat_ident.push((*__df_derive_v).clone());
            }
            #offsets_ident.push(#flat_ident.len());
        }
        if #flat_ident.is_empty() {
            #consume_empty
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_to_dataframe(&#flat_ident)?;
            #consume_filled
        }
    }}
}

/// Bulk emit for `payload: Vec<T>` where `T` is a concrete derived struct
/// type (i.e. has the inherent `__df_derive_columnar_from_refs` helper).
/// Flattens via `&T` references — no `T: Clone` requirement — and calls
/// the inherent helper directly, then ships each inner column to `ctx` via
/// `bulk_vec_consume_inner_columns`.
pub fn gen_bulk_concrete_vec(
    ty: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let consume_filled =
        bulk_vec_consume_inner_columns(ctx, &df_ident, &offsets_ident, &schema_iter);
    let consume_empty = bulk_vec_consume_empty_columns(ctx, &schema_iter);

    quote! {{
        let mut #flat_ident: ::std::vec::Vec<&#ty> = ::std::vec::Vec::new();
        let mut #offsets_ident: ::std::vec::Vec<usize> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_v in (&(#access)).iter() {
                #flat_ident.push(__df_derive_v);
            }
            #offsets_ident.push(#flat_ident.len());
        }
        if #flat_ident.is_empty() {
            #consume_empty
        } else {
            let #df_ident = #ty::__df_derive_columnar_from_refs(&#flat_ident)?;
            #consume_filled
        }
    }}
}
