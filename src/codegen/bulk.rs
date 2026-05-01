// Bulk emitters for nested-struct/generic-typed fields.
//
// These produce the entire builder/finisher token stream for a nested-typed
// field (the field's base type is either a derived struct or a type
// parameter of the enclosing struct). They sidestep the per-row push
// pipeline entirely by collecting `&T` references into a contiguous slice
// once and calling `<T as Columnar>::columnar_from_refs(&refs)` exactly
// once. From there each schema column of `T` is sliced/scattered into the
// parent's columns or `AnyValue::List` entries.
//
// Four wrapper shapes are supported as bulk: the bare leaf (`payload: T`),
// `Option<T>`, `Vec<T>`, and `Option<Vec<T>>`. The remaining nestings
// (`Vec<Option<T>>`, `Vec<Vec<T>>`, etc.) keep using the per-row trait-only
// paths defined elsewhere; those are rarer and the bulk variants would
// need additional position bookkeeping that isn't worth the added
// complexity.
//
// Generic and concrete shapes share the same emitter — the trait method
// `Columnar::columnar_from_refs(&[&Self])` is the one entry point for both,
// so neither path needs to clone elements into an owned `Vec<T>`.

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
    /// Builder-position emit inside `columnar_from_refs`. Pushes prefixed
    /// columns onto the in-scope `columns` Vec.
    Columnar { parent_name: &'a str },
    /// Finisher-position emit inside `__df_derive_vec_to_inner_list_values`.
    /// Pushes `AnyValue::List(inner)` onto the in-scope `out_values` Vec.
    VecAnyvalues,
}

/// Build the per-column emit body that adapts an inner `Series` to the given
/// context. `series_expr` must evaluate to an owned `polars::prelude::Series`.
///
/// Note: this helper assumes `<T as Columnar>::columnar_from_refs` returns a
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
            let __df_derive_prefixed = ::std::format!("{}.{}", #parent_name, #col_name_var);
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

/// Bulk emit for a leaf nested field (`payload: T`). Collects `Vec<&T>`
/// once, calls `<T as Columnar>::columnar_from_refs`, then ships each
/// schema column to `ctx`.
pub fn gen_bulk_leaf(
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let refs_ident = format_ident!("__df_derive_gen_refs_{}", idx);
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
        let #refs_ident: ::std::vec::Vec<&#ty> = items
            .iter()
            .map(|__df_derive_it| &(#access))
            .collect();
        let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#refs_ident)?;
        for (__df_derive_col_name, _) in <#ty as #to_df_trait>::schema()? {
            let __df_derive_col_name: &str = __df_derive_col_name.as_str();
            #consume
        }
    }}
}

/// Bulk emit for `payload: Option<T>`. Builds the gather (`nn`) plus
/// position (`pos`) vectors, calls `<T as Columnar>::columnar_from_refs`
/// once on the non-`None` references, then scatters each inner column back
/// over the original positions, emitting nulls where the source was `None`.
///
/// Three runtime branches:
/// - Every item is `None`: emit a typed-null inner for each column.
/// - Every item is `Some` (gather length matches `items.len()`): skip the
///   `IdxCa` scatter — `take` would be the identity here, and the setup
///   cost dominates on small inputs (especially N=1, where it would be
///   pure overhead vs. the per-row path).
/// - Mixed: build the gather `DataFrame`, then scatter each column with a
///   typed `take` over an `IdxCa` of original positions.
pub fn gen_bulk_option(
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

    let consume_direct = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {
            #df_ident
                .column(#col_name_var)?
                .as_materialized_series()
                .clone()
        },
    );
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
        let mut #nn_ident: ::std::vec::Vec<&#ty> = ::std::vec::Vec::new();
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for __df_derive_it in items {
            match &(#access) {
                ::std::option::Option::Some(__df_derive_v) => {
                    #pos_ident.push(::std::option::Option::Some(
                        #nn_ident.len() as #pp::IdxSize,
                    ));
                    #nn_ident.push(__df_derive_v);
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
        } else if #nn_ident.len() == items.len() {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#nn_ident)?;
            for (#col_name_var, _) in <#ty as #to_df_trait>::schema()? {
                let #col_name_var: &str = #col_name_var.as_str();
                #consume_direct
            }
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#nn_ident)?;
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

/// Build a `Series` expression that wraps `inner_col_expr` (a single-chunk
/// Series of inner values) as a `List<inner_logical_dtype>` outer column,
/// using `offsets_buf_expr` (a fully-validated `OffsetsBuffer<i64>`) to
/// partition the inner array into per-outer-row slices.
///
/// `validity_expr` is the token stream evaluating to
/// `Option<polars_arrow::bitmap::Bitmap>` — pass a literal
/// `::std::option::Option::None` for the non-`Option<Vec>` paths, or an
/// expression building a `Some(Bitmap)` whose length matches the number of
/// outer rows for the `Option<Vec>` path.
///
/// The result avoids the redundant copy that `ListBuilderTrait::append_series`
/// performs after `Inner::columnar_from_refs` already materialized the inner
/// values: we reuse the `Arc<dyn Array>` chunk straight from the inner Series
/// (`rechunk` + `chunks()[0].clone()` clones only the `Arc`, not the
/// element data) and stitch it under an `Arc::clone`-shared
/// `LargeListArray` view.
fn bulk_vec_emit_list_series(
    inner_col_expr: &TokenStream,
    offsets_buf_expr: &TokenStream,
    logical_inner_dtype_expr: &TokenStream,
    validity_expr: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    // The `unsafe` call to `Series::from_chunks_and_dtype_unchecked` is
    // hoisted into a free helper (`__df_derive_assemble_list_series_unchecked`)
    // emitted at the top of the per-derive `const _: () = { ... };` scope.
    // Keeping the unsafe outside the impl methods on `Self` prevents
    // `clippy::unsafe_derive_deserialize` from firing in downstream crates
    // that pair `#[derive(ToDataFrame)]` with `#[derive(Deserialize)]`.
    quote! {{
        let __df_derive_inner_col: #pp::Series = #inner_col_expr;
        let __df_derive_inner_rech = __df_derive_inner_col.rechunk();
        let __df_derive_inner_chunk: #pp::ArrayRef =
            __df_derive_inner_rech.chunks()[0].clone();
        let __df_derive_inner_arrow_dt = __df_derive_inner_chunk.dtype().clone();
        let __df_derive_list_arr = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(__df_derive_inner_arrow_dt),
            #offsets_buf_expr,
            __df_derive_inner_chunk,
            #validity_expr,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_list_arr,
            #logical_inner_dtype_expr,
        )
    }}
}

/// Build the per-inner-column emit for a bulk-vec context: given the inner
/// `DataFrame` (already computed from the flat slice) plus the shared
/// `OffsetsBuffer`, wrap the inner column under a single `LargeListArray`
/// whose offsets we already computed. The result is a `ListChunked` Series
/// that goes straight into either the parent `columns` Vec or an outer
/// `AnyValue::List`.
///
/// `validity_expr` is the per-call token stream for the outer-list validity
/// bitmap: `::std::option::Option::None` for the `[Vec]` path, or a
/// `Some(bitmap.clone())` expression for the `[Option, Vec]` path. The
/// bitmap is constructed once per field (above the schema loop) and cloned
/// per inner column so each `LargeListArray` carries its own
/// `Arc`-shared view.
///
/// Direct `LargeListArray` construction skips the redundant copy that
/// `ListBuilderTrait::append_series` performs after `Inner::columnar_from_refs`
/// already materialized the inner values. The chunk itself is an
/// `Arc<dyn Array>` clone of the inner Series's first chunk — element data
/// is not duplicated.
fn bulk_vec_consume_inner_columns(
    ctx: BulkContext<'_>,
    df_ident: &Ident,
    offsets_buf_ident: &Ident,
    schema_iter_ts: &TokenStream,
    validity_expr: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let inner_col_expr = quote! {
        #df_ident
            .column(#col_name_var)?
            .as_materialized_series()
            .clone()
    };
    let offsets_buf_expr = quote! { ::std::clone::Clone::clone(&#offsets_buf_ident) };
    let logical_dtype_expr = quote! { (*#dtype_var).clone() };
    let series_expr = bulk_vec_emit_list_series(
        &inner_col_expr,
        &offsets_buf_expr,
        &logical_dtype_expr,
        validity_expr,
    );
    let consume_filled = bulk_consume_inner_series(ctx, &col_name_var, &series_expr);
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
/// inner `columnar_from_refs` call), still produce one outer-list column
/// per inner schema entry, each containing `items.len()` empty inner lists.
///
/// Uses the same direct `LargeListArray` construction as the filled branch:
/// a single empty-Series chunk plus an all-zero offsets `Vec` produces the
/// correct `[List, List, ...]` shape where every row is a present-but-empty
/// inner list. (`validity = None` keeps the lists non-null; the
/// `Option<Vec>` path handles null lists.)
fn bulk_vec_consume_empty_columns(
    ctx: BulkContext<'_>,
    empty_offsets_buf_ident: &Ident,
    schema_iter_ts: &TokenStream,
    validity_expr: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let inner_col_expr = quote! {
        #pp::Series::new_empty("".into(), #dtype_var)
    };
    let offsets_buf_expr = quote! { ::std::clone::Clone::clone(&#empty_offsets_buf_ident) };
    let logical_dtype_expr = quote! { (*#dtype_var).clone() };
    let series_expr = bulk_vec_emit_list_series(
        &inner_col_expr,
        &offsets_buf_expr,
        &logical_dtype_expr,
        validity_expr,
    );
    let consume_empty = bulk_consume_inner_series(ctx, &col_name_var, &series_expr);
    quote! {
        for (#col_name_var, #dtype_var) in #schema_iter_ts {
            let #col_name_var: &str = #col_name_var.as_str();
            let #dtype_var: &#pp::DataType = &#dtype_var;
            #consume_empty
        }
    }
}

/// Bulk emit for `payload: Vec<T>`. Flattens via `&T` references — no
/// `T: Clone` requirement — and calls `<T as Columnar>::columnar_from_refs`
/// once, then ships each inner column to `ctx` via
/// `bulk_vec_consume_inner_columns`.
///
/// `pa_root` is the cached token stream for the `polars-arrow` crate root,
/// resolved once per macro invocation by the caller and threaded through
/// here so we don't re-run `proc_macro_crate::crate_name` per field.
pub fn gen_bulk_vec(
    pa_root: &TokenStream,
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let offsets_buf_ident = format_ident!("__df_derive_gen_offsets_buf_{}", idx);
    let empty_offsets_buf_ident = format_ident!("__df_derive_gen_empty_offsets_buf_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let no_validity = quote! { ::std::option::Option::None };
    let consume_filled = bulk_vec_consume_inner_columns(
        ctx,
        &df_ident,
        &offsets_buf_ident,
        &schema_iter,
        &no_validity,
    );
    let consume_empty =
        bulk_vec_consume_empty_columns(ctx, &empty_offsets_buf_ident, &schema_iter, &no_validity);

    // Build the shared `OffsetsBuffer` once per branch — every inner-column
    // iteration clones the buffer (cheap; `OffsetsBuffer` wraps an
    // `Arc`-shared `Buffer`).
    let filled_buf_setup = quote! {
        let #offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#offsets_ident)?;
    };
    let empty_buf_setup = quote! {
        let #empty_offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(::std::vec![0i64; items.len() + 1])?;
    };

    quote! {{
        let mut #flat_ident: ::std::vec::Vec<&#ty> = ::std::vec::Vec::new();
        let mut #offsets_ident: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_v in (&(#access)).iter() {
                #flat_ident.push(__df_derive_v);
            }
            #offsets_ident.push(#flat_ident.len() as i64);
        }
        if #flat_ident.is_empty() {
            #empty_buf_setup
            #consume_empty
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#flat_ident)?;
            #filled_buf_setup
            #consume_filled
        }
    }}
}

/// Bulk emit for `payload: Option<Vec<T>>`. Like `gen_bulk_vec` but each
/// parent row is either `Some(Vec<T>)` (contributes inner refs and an
/// offset entry, marked valid in the outer-list bitmap) or `None`
/// (contributes no inner refs, repeats the previous offset, marked null).
/// `Inner::columnar_from_refs` runs exactly once over the flattened slice;
/// the resulting outer `LargeListArray` carries a `Bitmap` so `None`-rows
/// surface as null lists rather than empty ones.
///
/// `pa_root` is the cached token stream for the `polars-arrow` crate root.
pub fn gen_bulk_option_vec(
    pa_root: &TokenStream,
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let offsets_buf_ident = format_ident!("__df_derive_gen_offsets_buf_{}", idx);
    let empty_offsets_buf_ident = format_ident!("__df_derive_gen_empty_offsets_buf_{}", idx);
    let validity_bitmap_ident = format_ident!("__df_derive_gen_validity_bm_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    // Each inner-column iteration clones the bitmap into the
    // `LargeListArray`; `Bitmap` is `Arc`-shared so this is cheap.
    let validity_expr = quote! {
        ::std::option::Option::Some(::std::clone::Clone::clone(&#validity_bitmap_ident))
    };
    let consume_filled = bulk_vec_consume_inner_columns(
        ctx,
        &df_ident,
        &offsets_buf_ident,
        &schema_iter,
        &validity_expr,
    );
    let consume_empty =
        bulk_vec_consume_empty_columns(ctx, &empty_offsets_buf_ident, &schema_iter, &validity_expr);

    // Build the shared `OffsetsBuffer` once per branch.
    let filled_buf_setup = quote! {
        let #offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#offsets_ident)?;
    };
    let empty_buf_setup = quote! {
        let #empty_offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(::std::vec![0i64; items.len() + 1])?;
    };

    // The validity bitmap is built once from a `MutableBitmap` populated in
    // the same outer-row scan that builds offsets/flat refs, then frozen
    // via `From<MutableBitmap> for Bitmap`. Pre-sizing with
    // `with_capacity(items.len())` skips the otherwise-amortized regrowth.
    quote! {{
        let mut #flat_ident: ::std::vec::Vec<&#ty> = ::std::vec::Vec::new();
        let mut #offsets_ident: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        let mut __df_derive_validity_mb: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
        for __df_derive_it in items {
            match &(#access) {
                ::std::option::Option::Some(__df_derive_inner_vec) => {
                    for __df_derive_v in __df_derive_inner_vec.iter() {
                        #flat_ident.push(__df_derive_v);
                    }
                    __df_derive_validity_mb.push(true);
                }
                ::std::option::Option::None => {
                    __df_derive_validity_mb.push(false);
                }
            }
            #offsets_ident.push(#flat_ident.len() as i64);
        }
        let #validity_bitmap_ident: #pa_root::bitmap::Bitmap =
            <#pa_root::bitmap::Bitmap as ::core::convert::From<
                #pa_root::bitmap::MutableBitmap,
            >>::from(__df_derive_validity_mb);
        if #flat_ident.is_empty() {
            #empty_buf_setup
            #consume_empty
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#flat_ident)?;
            #filled_buf_setup
            #consume_filled
        }
    }}
}
