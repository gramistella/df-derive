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
// Seven wrapper shapes are supported as bulk: the bare leaf (`payload: T`),
// `Option<T>`, `Vec<T>`, `Option<Vec<T>>`, `Vec<Option<T>>`,
// `Option<Vec<Option<T>>>`, and `Vec<Vec<T>>`. The remaining nestings
// (`Option<Option<T>>`, deeper triple-Vec shapes, etc.) keep using the
// per-row trait-only paths defined elsewhere; those are rarer and the bulk
// variants would need additional position bookkeeping that isn't worth the
// added complexity.
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

/// Build a `Series` expression that wraps `inner_col_expr` under **two**
/// stacked `LargeListArray` layers: the inner partitions leaf values into
/// per-inner-list slices via `inner_offsets_buf_expr`, and the outer groups
/// inner lists into per-outer-row slices via `outer_offsets_buf_expr`.
///
/// Used by the `[Vec, Vec]` bulk emitter for nested-struct/generic shapes —
/// the runtime dtype of the resulting Series is `List<List<leaf>>` even
/// though the schema entry only reports `List<leaf>` (a pre-existing
/// limitation of `generate_schema_entries_for_struct` shared with the slow
/// path; see the assertion in `tests/pass/20-generics.rs`).
///
/// `logical_inner_leaf_dtype_expr` is the leaf dtype (e.g. `DataType::Int64`).
/// The free helper `__df_derive_assemble_list_series_unchecked` wraps it in
/// one more `List<>` layer for the final Series's logical dtype.
fn bulk_vec_emit_double_list_series(
    inner_col_expr: &TokenStream,
    inner_offsets_buf_expr: &TokenStream,
    outer_offsets_buf_expr: &TokenStream,
    logical_inner_leaf_dtype_expr: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    quote! {{
        let __df_derive_inner_col: #pp::Series = #inner_col_expr;
        let __df_derive_inner_rech = __df_derive_inner_col.rechunk();
        let __df_derive_inner_chunk: #pp::ArrayRef =
            __df_derive_inner_rech.chunks()[0].clone();
        let __df_derive_inner_arrow_dt = __df_derive_inner_chunk.dtype().clone();
        let __df_derive_inner_list_arr = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(__df_derive_inner_arrow_dt),
            #inner_offsets_buf_expr,
            __df_derive_inner_chunk,
            ::std::option::Option::None,
        );
        let __df_derive_inner_list_chunk: #pp::ArrayRef =
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef;
        let __df_derive_inner_list_arrow_dt =
            __df_derive_inner_list_chunk.dtype().clone();
        let __df_derive_outer_list_arr = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(__df_derive_inner_list_arrow_dt),
            #outer_offsets_buf_expr,
            __df_derive_inner_list_chunk,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #pp::DataType::List(::std::boxed::Box::new(#logical_inner_leaf_dtype_expr)),
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
    let col_name_var = quote! { __df_derive_col_name };
    let inner_col_expr = quote! {
        #df_ident
            .column(#col_name_var)?
            .as_materialized_series()
            .clone()
    };
    bulk_vec_consume_columns_with_expr(
        ctx,
        offsets_buf_ident,
        schema_iter_ts,
        validity_expr,
        &inner_col_expr,
    )
}

/// Per-inner-column consume loop sharing the `LargeListArray` assembly with
/// `bulk_vec_consume_inner_columns`, but parameterized over the inner-Series
/// expression. Lets the `[Vec, Option]` path swap in a `take`-expanded inner
/// Series without copying the unsafe-hoisting plumbing.
fn bulk_vec_consume_columns_with_expr(
    ctx: BulkContext<'_>,
    offsets_buf_ident: &Ident,
    schema_iter_ts: &TokenStream,
    validity_expr: &TokenStream,
    inner_col_expr: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let offsets_buf_expr = quote! { ::std::clone::Clone::clone(&#offsets_buf_ident) };
    let logical_dtype_expr = quote! { (*#dtype_var).clone() };
    let series_expr = bulk_vec_emit_list_series(
        inner_col_expr,
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
    let total_ident = format_ident!("__df_derive_gen_total_{}", idx);
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
        let mut #total_ident: usize = 0;
        for __df_derive_it in items {
            #total_ident += (&(#access)).len();
        }
        let mut #flat_ident: ::std::vec::Vec<&#ty> =
            ::std::vec::Vec::with_capacity(#total_ident);
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
    let total_ident = format_ident!("__df_derive_gen_total_{}", idx);
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
        let mut #total_ident: usize = 0;
        for __df_derive_it in items {
            if let ::std::option::Option::Some(__df_derive_inner_vec) = &(#access) {
                #total_ident += __df_derive_inner_vec.len();
            }
        }
        let mut #flat_ident: ::std::vec::Vec<&#ty> =
            ::std::vec::Vec::with_capacity(#total_ident);
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

/// Bundle of per-branch consume tokens for `gen_bulk_vec_option` and
/// `gen_bulk_option_vec_option`. Pre-built before the `quote!` body so the
/// main function stays focused on the outer-row scan and branch dispatch.
struct VecOptionConsumes {
    direct: TokenStream,
    filled: TokenStream,
    all_absent: TokenStream,
    empty: TokenStream,
}

/// Identifiers shared between the consume-token builder and the emitted
/// outer-row scan. Bundled into a struct so `build_vec_option_consumes` and
/// `gen_bulk_option_vec_option` (which both reference all five) keep their
/// argument counts under the clippy limit.
struct VecOptionIdents {
    df: Ident,
    offsets_buf: Ident,
    empty_offsets_buf: Ident,
    take: Ident,
    total: Ident,
}

impl VecOptionIdents {
    fn new(idx: usize) -> Self {
        Self {
            df: format_ident!("__df_derive_gen_df_{}", idx),
            offsets_buf: format_ident!("__df_derive_gen_offsets_buf_{}", idx),
            empty_offsets_buf: format_ident!("__df_derive_gen_empty_offsets_buf_{}", idx),
            take: format_ident!("__df_derive_gen_take_{}", idx),
            total: format_ident!("__df_derive_gen_total_{}", idx),
        }
    }
}

fn build_vec_option_consumes(
    ctx: BulkContext<'_>,
    idents: &VecOptionIdents,
    schema_iter: &TokenStream,
    validity_expr: &TokenStream,
) -> VecOptionConsumes {
    let pp = super::polars_paths::prelude();
    let VecOptionIdents {
        df: df_ident,
        offsets_buf: offsets_buf_ident,
        empty_offsets_buf: empty_offsets_buf_ident,
        take: take_ident,
        total: total_ident,
    } = idents;
    let direct = bulk_vec_consume_inner_columns(
        ctx,
        df_ident,
        offsets_buf_ident,
        schema_iter,
        validity_expr,
    );
    let take_expr = quote! {{
        let __df_derive_inner_full = #df_ident
            .column(__df_derive_col_name)?
            .as_materialized_series();
        __df_derive_inner_full.take(&#take_ident)?
    }};
    let filled = bulk_vec_consume_columns_with_expr(
        ctx,
        offsets_buf_ident,
        schema_iter,
        validity_expr,
        &take_expr,
    );
    let null_expr = quote! {
        #pp::Series::new_empty("".into(), __df_derive_dtype)
            .extend_constant(#pp::AnyValue::Null, #total_ident)?
    };
    let all_absent = bulk_vec_consume_columns_with_expr(
        ctx,
        offsets_buf_ident,
        schema_iter,
        validity_expr,
        &null_expr,
    );
    let empty =
        bulk_vec_consume_empty_columns(ctx, empty_offsets_buf_ident, schema_iter, validity_expr);
    VecOptionConsumes {
        direct,
        filled,
        all_absent,
        empty,
    }
}

/// Emits the two `OffsetsBuffer` setup statements (filled + empty) shared by
/// the `[Vec, Option]` and `[Option, Vec, Option]` paths. The filled buffer
/// consumes a pre-built `Vec<i64>`; the empty buffer is `vec![0; n+1]` so the
/// `total == 0` branch can still wrap empty inner Series under
/// `LargeListArray` with the correct outer-row count.
fn vec_option_buf_setups(
    pa_root: &TokenStream,
    offsets_ident: &Ident,
    offsets_buf_ident: &Ident,
    empty_offsets_buf_ident: &Ident,
) -> (TokenStream, TokenStream) {
    let filled = quote! {
        let #offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#offsets_ident)?;
    };
    let empty = quote! {
        let #empty_offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(::std::vec![0i64; items.len() + 1])?;
    };
    (filled, empty)
}

/// Bulk emit for `payload: Vec<Option<T>>`. Inverse of `gen_bulk_option_vec`:
/// the **outer** list is always non-null, but **inner elements** can be
/// null. Per parent row we iterate the outer Vec, splitting `Some(v)`
/// references into a flat slice and recording per-element positions:
/// `Some(j)` for a present element (where `j` indexes the gathered inner
/// `DataFrame`) and `None` for an absent one.
///
/// `Inner::columnar_from_refs` runs once over the gathered slice. For each
/// inner schema column we then call `Series::take(&IdxCa)` against the
/// per-element positions to expand the gathered column back to the
/// total-element count, with null entries materialized at the `None`
/// positions. The resulting per-column Series is wrapped in a
/// `LargeListArray` with `validity = None` (outer list rows are non-null)
/// and the parent-row offsets. The single `take` per inner schema column
/// replaces the per-parent-row `take` performed by the slow
/// `nested.rs::gen_nested_vec_anyvalues_option` path.
///
/// Four runtime branches:
/// - `total == 0`: every outer Vec is empty, so emit one outer-list column
///   per inner schema entry of `items.len()` empty (non-null) lists.
/// - `flat.is_empty()` (but `total > 0`): every Some-position turned out to
///   be None, so the gathered slice is empty. Skip the
///   `columnar_from_refs(&[])` round-trip and emit per-column null Series
///   of length `total` directly.
/// - `flat.len() == total`: no inner nulls — gathered Series already has
///   the right length. Skip the `IdxCa` + `take` overhead.
/// - Mixed: build the `IdxCa` once and `take` per inner column.
///
/// `pa_root` is the cached token stream for the `polars-arrow` crate root.
pub fn gen_bulk_vec_option(
    pa_root: &TokenStream,
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let pos_ident = format_ident!("__df_derive_gen_pos_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let idents = VecOptionIdents::new(idx);
    let VecOptionIdents {
        df: df_ident,
        offsets_buf: offsets_buf_ident,
        empty_offsets_buf: empty_offsets_buf_ident,
        take: take_ident,
        total: total_ident,
    } = &idents;
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let no_validity = quote! { ::std::option::Option::None };
    let VecOptionConsumes {
        direct: consume_direct,
        filled: consume_filled,
        all_absent: consume_all_absent,
        empty: consume_empty,
    } = build_vec_option_consumes(ctx, &idents, &schema_iter, &no_validity);
    let (filled_buf_setup, empty_buf_setup) = vec_option_buf_setups(
        pa_root,
        &offsets_ident,
        offsets_buf_ident,
        empty_offsets_buf_ident,
    );

    quote! {{
        let mut #total_ident: usize = 0;
        for __df_derive_it in items {
            #total_ident += (&(#access)).len();
        }
        let mut #flat_ident: ::std::vec::Vec<&#ty> =
            ::std::vec::Vec::with_capacity(#total_ident);
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(#total_ident);
        let mut #offsets_ident: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_maybe in (&(#access)).iter() {
                match __df_derive_maybe {
                    ::std::option::Option::Some(__df_derive_v) => {
                        #pos_ident.push(::std::option::Option::Some(
                            #flat_ident.len() as #pp::IdxSize,
                        ));
                        #flat_ident.push(__df_derive_v);
                    }
                    ::std::option::Option::None => {
                        #pos_ident.push(::std::option::Option::None);
                    }
                }
            }
            #offsets_ident.push(#pos_ident.len() as i64);
        }
        if #total_ident == 0 {
            #empty_buf_setup
            #consume_empty
        } else if #flat_ident.is_empty() {
            #filled_buf_setup
            #consume_all_absent
        } else if #flat_ident.len() == #total_ident {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#flat_ident)?;
            #filled_buf_setup
            #consume_direct
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#flat_ident)?;
            let #take_ident: #pp::IdxCa =
                <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #pos_ident.iter().copied(),
                );
            #filled_buf_setup
            #consume_filled
        }
    }}
}

/// Bulk emit for `payload: Option<Vec<Option<T>>>`. Fuses the validity-bitmap
/// outer-list pattern from `gen_bulk_option_vec` with the per-element scatter
/// from `gen_bulk_vec_option`. Per parent row the outer scan branches on the
/// outer `Option`:
/// - `Some(inner_vec)`: walk `inner_vec`, splitting `Some(v)` references into
///   `flat` (with an `IdxSize` position appended to `pos`) or pushing
///   `pos.push(None)` for null inner elements; the outer validity bit is
///   `true` and the offset advances by `inner_vec.len()`.
/// - `None`: outer validity bit is `false`, the offset repeats the previous
///   value (delta = 0), `pos`/`flat` are untouched.
///
/// The bitmap is frozen once and `Arc`-shared across every inner schema
/// column. The same four-branch dispatch as `gen_bulk_vec_option` applies,
/// just with the bitmap threaded through the `validity_expr`:
/// - `total == 0`: every Some inner Vec was empty (or every parent was
///   None) — empty offsets buffer + zero-length inner Series + bitmap.
/// - `flat.is_empty() && total > 0`: every inner element was None — null
///   inner Series of length `total` + actual offsets + bitmap.
/// - `flat.len() == total`: no inner nulls — direct gather + actual offsets
///   + bitmap, skipping the `IdxCa` scatter.
/// - mixed: gather + `IdxCa` scatter per inner column + bitmap.
pub fn gen_bulk_option_vec_option(
    pa_root: &TokenStream,
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let pos_ident = format_ident!("__df_derive_gen_pos_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let validity_bitmap_ident = format_ident!("__df_derive_gen_validity_bm_{}", idx);
    let idents = VecOptionIdents::new(idx);
    let VecOptionIdents {
        df: df_ident,
        offsets_buf: offsets_buf_ident,
        empty_offsets_buf: empty_offsets_buf_ident,
        take: take_ident,
        total: total_ident,
    } = &idents;
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let validity_expr = quote! {
        ::std::option::Option::Some(::std::clone::Clone::clone(&#validity_bitmap_ident))
    };
    let VecOptionConsumes {
        direct: consume_direct,
        filled: consume_filled,
        all_absent: consume_all_absent,
        empty: consume_empty,
    } = build_vec_option_consumes(ctx, &idents, &schema_iter, &validity_expr);
    let (filled_buf_setup, empty_buf_setup) = vec_option_buf_setups(
        pa_root,
        &offsets_ident,
        offsets_buf_ident,
        empty_offsets_buf_ident,
    );
    let scan = option_vec_option_scan(
        pa_root,
        access,
        ScanIdents {
            flat: &flat_ident,
            pos: &pos_ident,
            offsets: &offsets_ident,
            total: total_ident,
            validity_bitmap: &validity_bitmap_ident,
        },
        ty,
    );

    quote! {{
        let mut #total_ident: usize = 0;
        for __df_derive_it in items {
            if let ::std::option::Option::Some(__df_derive_inner_vec) = &(#access) {
                #total_ident += __df_derive_inner_vec.len();
            }
        }
        #scan
        if #total_ident == 0 {
            #empty_buf_setup
            #consume_empty
        } else if #flat_ident.is_empty() {
            #filled_buf_setup
            #consume_all_absent
        } else if #flat_ident.len() == #total_ident {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#flat_ident)?;
            #filled_buf_setup
            #consume_direct
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#flat_ident)?;
            let #take_ident: #pp::IdxCa =
                <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #pos_ident.iter().copied(),
                );
            #filled_buf_setup
            #consume_filled
        }
    }}
}

/// Build the per-inner-column emit for the populated `[Vec, Vec]` bulk
/// path: pull each inner schema column out of the inner `DataFrame` and
/// stack two `LargeListArray` layers (inner partitions leaves into per-
/// inner-list slices, outer groups inner lists into per-outer-row slices).
///
/// Unlike `bulk_vec_consume_inner_columns`, this builds a `List<List<…>>`
/// runtime Series; the schema entry only declares `List<…>` (the existing
/// limitation matched by the slow path), so the assemble helper's outer
/// `List<>` wrap surfaces only at runtime.
fn bulk_vec_vec_consume_inner_columns(
    ctx: BulkContext<'_>,
    df_ident: &Ident,
    inner_offsets_buf_ident: &Ident,
    outer_offsets_buf_ident: &Ident,
    schema_iter_ts: &TokenStream,
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
    let inner_offsets_buf_expr = quote! { ::std::clone::Clone::clone(&#inner_offsets_buf_ident) };
    let outer_offsets_buf_expr = quote! { ::std::clone::Clone::clone(&#outer_offsets_buf_ident) };
    let logical_inner_leaf_dtype_expr = quote! { (*#dtype_var).clone() };
    let series_expr = bulk_vec_emit_double_list_series(
        &inner_col_expr,
        &inner_offsets_buf_expr,
        &outer_offsets_buf_expr,
        &logical_inner_leaf_dtype_expr,
    );
    let consume = bulk_consume_inner_series(ctx, &col_name_var, &series_expr);
    quote! {
        for (#col_name_var, #dtype_var) in #schema_iter_ts {
            let #col_name_var: &str = #col_name_var.as_str();
            let #dtype_var: &#pp::DataType = &#dtype_var;
            #consume
        }
    }
}

/// Build the empty-flat emit for the `[Vec, Vec]` bulk path: when no leaves
/// exist (every inner Vec is empty across every outer row), still produce
/// one outer-list column per inner schema entry. Each Series uses the real
/// `inner_offsets_buf` and `outer_offsets_buf` (so the per-row inner-list
/// counts and per-inner-list zero leaf counts are preserved) plus an empty
/// inner Series of the leaf dtype.
fn bulk_vec_vec_consume_empty_columns(
    ctx: BulkContext<'_>,
    inner_offsets_buf_ident: &Ident,
    outer_offsets_buf_ident: &Ident,
    schema_iter_ts: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let inner_col_expr = quote! {
        #pp::Series::new_empty("".into(), #dtype_var)
    };
    let inner_offsets_buf_expr = quote! { ::std::clone::Clone::clone(&#inner_offsets_buf_ident) };
    let outer_offsets_buf_expr = quote! { ::std::clone::Clone::clone(&#outer_offsets_buf_ident) };
    let logical_inner_leaf_dtype_expr = quote! { (*#dtype_var).clone() };
    let series_expr = bulk_vec_emit_double_list_series(
        &inner_col_expr,
        &inner_offsets_buf_expr,
        &outer_offsets_buf_expr,
        &logical_inner_leaf_dtype_expr,
    );
    let consume = bulk_consume_inner_series(ctx, &col_name_var, &series_expr);
    quote! {
        for (#col_name_var, #dtype_var) in #schema_iter_ts {
            let #col_name_var: &str = #col_name_var.as_str();
            let #dtype_var: &#pp::DataType = &#dtype_var;
            #consume
        }
    }
}

/// Bulk emit for `payload: Vec<Vec<T>>` over a nested struct or generic base.
/// Mirrors `gen_bulk_vec` but stacks two `LargeListArray`s: leaves are
/// flattened across both axes, with `inner_offsets` recording per-inner-list
/// leaf counts and `outer_offsets` recording per-outer-row inner-list counts.
/// `Inner::columnar_from_refs` runs exactly once on the flat leaf slice; each
/// inner schema column is then wrapped twice (inner-list partition + outer-
/// list group) before shipping to `ctx`.
///
/// The runtime Series carries dtype `List<List<leaf>>` while the schema
/// entry declares only `List<leaf>` — same convention as the slow path
/// (see `tests/pass/20-generics.rs` line 826).
///
/// `pa_root` is the cached token stream for the `polars-arrow` crate root.
pub fn gen_bulk_vec_vec(
    pa_root: &TokenStream,
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let inner_offsets_ident = format_ident!("__df_derive_gen_inner_offsets_{}", idx);
    let outer_offsets_ident = format_ident!("__df_derive_gen_outer_offsets_{}", idx);
    let inner_offsets_buf_ident = format_ident!("__df_derive_gen_inner_offsets_buf_{}", idx);
    let outer_offsets_buf_ident = format_ident!("__df_derive_gen_outer_offsets_buf_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let total_inners_ident = format_ident!("__df_derive_gen_total_inners_{}", idx);
    let total_leaves_ident = format_ident!("__df_derive_gen_total_leaves_{}", idx);
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let consume_filled = bulk_vec_vec_consume_inner_columns(
        ctx,
        &df_ident,
        &inner_offsets_buf_ident,
        &outer_offsets_buf_ident,
        &schema_iter,
    );
    let consume_empty = bulk_vec_vec_consume_empty_columns(
        ctx,
        &inner_offsets_buf_ident,
        &outer_offsets_buf_ident,
        &schema_iter,
    );

    // The inner and outer offsets buffers are shared across the schema
    // iteration (cloned per inner column; `OffsetsBuffer` wraps an
    // `Arc`-shared `Buffer`). Both branches need both buffers — the empty
    // branch skips the `columnar_from_refs` call but still wraps an empty
    // inner Series under the same outer-list shape.
    let buf_setup = quote! {
        let #inner_offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#inner_offsets_ident)?;
        let #outer_offsets_buf_ident: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#outer_offsets_ident)?;
    };

    quote! {{
        let mut #total_inners_ident: usize = 0;
        let mut #total_leaves_ident: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner_vec in (&(#access)).iter() {
                #total_inners_ident += 1;
                #total_leaves_ident += __df_derive_inner_vec.len();
            }
        }
        let mut #flat_ident: ::std::vec::Vec<&#ty> =
            ::std::vec::Vec::with_capacity(#total_leaves_ident);
        let mut #inner_offsets_ident: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(#total_inners_ident + 1);
        #inner_offsets_ident.push(0);
        let mut #outer_offsets_ident: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #outer_offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_inner_vec in (&(#access)).iter() {
                for __df_derive_v in __df_derive_inner_vec.iter() {
                    #flat_ident.push(__df_derive_v);
                }
                #inner_offsets_ident.push(#flat_ident.len() as i64);
            }
            #outer_offsets_ident.push((#inner_offsets_ident.len() - 1) as i64);
        }
        if #flat_ident.is_empty() {
            #buf_setup
            #consume_empty
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_from_refs(&#flat_ident)?;
            #buf_setup
            #consume_filled
        }
    }}
}

/// Identifiers used by `option_vec_option_scan`. Bundled to keep the
/// argument count under the clippy limit.
#[derive(Clone, Copy)]
struct ScanIdents<'a> {
    flat: &'a Ident,
    pos: &'a Ident,
    offsets: &'a Ident,
    total: &'a Ident,
    validity_bitmap: &'a Ident,
}

/// Single-pass outer-row scan for `gen_bulk_option_vec_option`. Walks each
/// parent row's `Option<Vec<Option<T>>>`, populating `flat`, `pos`, the
/// per-row offset deltas, and the outer-list `MutableBitmap` (frozen at the
/// end of the scan into a shared `Bitmap`). `total` is computed in a
/// separate prior pass so `flat`/`pos` can be `with_capacity`-pre-sized.
fn option_vec_option_scan(
    pa_root: &TokenStream,
    access: &TokenStream,
    idents: ScanIdents<'_>,
    ty: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let ScanIdents {
        flat: flat_ident,
        pos: pos_ident,
        offsets: offsets_ident,
        total: total_ident,
        validity_bitmap: validity_bitmap_ident,
    } = idents;
    quote! {
        let mut #flat_ident: ::std::vec::Vec<&#ty> =
            ::std::vec::Vec::with_capacity(#total_ident);
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(#total_ident);
        let mut #offsets_ident: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        let mut __df_derive_validity_mb: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
        for __df_derive_it in items {
            match &(#access) {
                ::std::option::Option::Some(__df_derive_inner_vec) => {
                    for __df_derive_maybe in __df_derive_inner_vec.iter() {
                        match __df_derive_maybe {
                            ::std::option::Option::Some(__df_derive_v) => {
                                #pos_ident.push(::std::option::Option::Some(
                                    #flat_ident.len() as #pp::IdxSize,
                                ));
                                #flat_ident.push(__df_derive_v);
                            }
                            ::std::option::Option::None => {
                                #pos_ident.push(::std::option::Option::None);
                            }
                        }
                    }
                    __df_derive_validity_mb.push(true);
                }
                ::std::option::Option::None => {
                    __df_derive_validity_mb.push(false);
                }
            }
            #offsets_ident.push(#pos_ident.len() as i64);
        }
        let #validity_bitmap_ident: #pa_root::bitmap::Bitmap =
            <#pa_root::bitmap::Bitmap as ::core::convert::From<
                #pa_root::bitmap::MutableBitmap,
            >>::from(__df_derive_validity_mb);
    }
}
