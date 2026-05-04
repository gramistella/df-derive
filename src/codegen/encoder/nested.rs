//! Nested-struct/generic encoder paths (`CollectThenBulk` leaves).
//!
//! Ports the seven nested-struct/generic shapes (`[]`, `[Option]`,
//! `[Vec]`, `[Option, Vec]`, `[Vec, Option]`, `[Option, Vec, Option]`,
//! `[Vec, Vec]`) into the encoder IR. Each shape is built up from a single
//! `CollectThenBulk` leaf (which knows how to call
//! `<T as Columnar>::columnar_from_refs(&refs)`) plus the wrapper-stack-shaped
//! gather/scatter machinery in this module.
//!
//! The invariant: every `LargeListArray::new` routes through the in-scope free
//! helper `__df_derive_assemble_list_series_unchecked` (defined at the top of
//! each derive's `const _: () = { ... };` scope), keeping `unsafe` out of any
//! `Self`-bearing impl method so `clippy::unsafe_derive_deserialize` stays
//! silent on downstream `#[derive(ToDataFrame, Deserialize)]` types.
//!
//! Every shape produces an `Encoder::Multi { columnar }` because the inner
//! `DataFrame` carries one column per inner schema entry of `T`. The block
//! pushes one Series per inner schema column onto the call site's `columns`
//! vec, with the parent name prefixed onto each inner column name.

use crate::ir::Wrapper;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::shape_walk::{ScanLayerIdents, ShapeScan, shape_offsets_decls, shape_validity_decls};
use super::{Encoder, VecShape, WrapperKind, collapse_options_to_ref, normalize_wrappers};

/// Per-call-site context for nested-struct/generic encoders. Carries the
/// `polars-arrow` crate root (so the combinators don't re-resolve it per
/// call) plus the type-as-path expression and the fully-qualified trait
/// paths used in UFCS calls (`<#ty as #columnar_trait>::columnar_from_refs`,
/// `<#ty as #to_df_trait>::schema`).
pub struct NestedLeafCtx<'a> {
    pub access: &'a TokenStream,
    pub idx: usize,
    pub parent_name: &'a str,
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a TokenStream,
    pub to_df_trait: &'a TokenStream,
    pub pa_root: &'a TokenStream,
}

/// Per-shape identifier bundle for the nested encoder paths. Computing these
/// once at the top of each shape builder keeps the per-shape body focused on
/// the gather/scatter logic.
struct NestedIdents {
    /// `Vec<&T>` flat ref accumulator.
    flat: syn::Ident,
    /// `Vec<Option<IdxSize>>` per-element positions for the inner-Option
    /// scatter case.
    positions: syn::Ident,
    /// Inner `DataFrame` returned by `columnar_from_refs`.
    df: syn::Ident,
    /// `IdxCa` built from `positions` for the scatter case.
    take: syn::Ident,
    /// Total inner-element count (used by precount + outer-list capacity).
    total: syn::Ident,
}

impl NestedIdents {
    fn new(idx: usize) -> Self {
        Self {
            flat: format_ident!("__df_derive_gen_flat_{}", idx),
            positions: format_ident!("__df_derive_gen_pos_{}", idx),
            df: format_ident!("__df_derive_gen_df_{}", idx),
            take: format_ident!("__df_derive_gen_take_{}", idx),
            total: format_ident!("__df_derive_gen_total_{}", idx),
        }
    }
}

/// Build the per-column emit body that iterates `<T as ToDataFrame>::schema()?`
/// and pushes each inner-Series-yielding expression onto `columns` with the
/// parent name prefixed. The schema name is exposed as `__df_derive_col_name:
/// &str` and the dtype as `__df_derive_dtype: &polars::DataType` so per-column
/// expressions can reference both.
fn nested_consume_columns(
    parent_name: &str,
    to_df_trait: &TokenStream,
    ty: &TokenStream,
    series_expr: &TokenStream,
) -> TokenStream {
    let pp = crate::codegen::polars_paths::prelude();
    quote! {
        for (__df_derive_col_name, __df_derive_dtype) in
            <#ty as #to_df_trait>::schema()?
        {
            let __df_derive_col_name: &str = __df_derive_col_name.as_str();
            let __df_derive_dtype: &#pp::DataType = &__df_derive_dtype;
            {
                let __df_derive_prefixed = ::std::format!(
                    "{}.{}", #parent_name, __df_derive_col_name,
                );
                let __df_derive_inner: #pp::Series = #series_expr;
                let __df_derive_named = __df_derive_inner
                    .with_name(__df_derive_prefixed.as_str().into());
                columns.push(__df_derive_named.into());
            }
        }
    }
}

/// Build the bare-leaf nested encoder (`payload: T`). Gathers refs into
/// `Vec<&T>`, calls `columnar_from_refs` once, and per inner schema column
/// pulls the materialized `Series` straight out of the resulting `DataFrame`
/// (no list-array wrapping; the parent column is the inner column).
fn nested_leaf_encoder(ctx: &NestedLeafCtx<'_>) -> Encoder {
    let NestedLeafCtx {
        access,
        idx,
        parent_name,
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let df = &ids.df;
    let inner_expr = quote! {
        #df.column(__df_derive_col_name)?
            .as_materialized_series()
            .clone()
    };
    let columnar = nested_consume_columns(parent_name, to_df_trait, ty, &inner_expr);
    let setup = quote! {
        let #flat: ::std::vec::Vec<&#ty> = items
            .iter()
            .map(|__df_derive_it| &(#access))
            .collect();
        let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
    };
    let columnar_block = quote! {{ #setup #columnar }};
    Encoder::Multi {
        columnar: columnar_block,
    }
}

/// `option(nested_leaf)` — `[Option]` (or any consecutive run of `Option`s
/// over a struct/generic, since Polars folds nested Nones into one validity
/// bit). For `option_layers >= 2`, the caller pre-collapses the access into
/// an `Option<&T>` value-expression; the scan then reads the value directly
/// without a `&`.
///
/// Splits each row's `Option<T>` into a flat ref slice plus a
/// `Vec<Option<IdxSize>>` of positions. Three runtime branches:
/// - all None: emit one typed-null Series of length `items.len()` per inner
///   schema column.
/// - all Some (no scatter needed): pull each column straight from the inner
///   `DataFrame`, no `take`.
/// - mixed: build an `IdxCa` over positions and `take` per inner column to
///   scatter values back over the original row positions.
fn nested_option_encoder_collapsed(ctx: &NestedLeafCtx<'_>, option_layers: usize) -> Encoder {
    // For `option_layers >= 2`, `#access` is an `as_ref().and_then(...)`
    // chain returning `Option<&T>` directly — we match it by value.
    // For `option_layers == 1`, `#access` is the raw `&Option<T>` field
    // expression — we match by reference. The two arms produce slightly
    // different scans because the bound `__df_derive_v` is `&T` either way,
    // but the surrounding match expression differs.
    let access_ts = ctx.access.clone();
    let match_expr = if option_layers >= 2 {
        quote! { (#access_ts) }
    } else {
        quote! { &(#access_ts) }
    };
    nested_option_encoder_impl(ctx, &match_expr)
}

fn nested_option_encoder_impl(ctx: &NestedLeafCtx<'_>, match_expr: &TokenStream) -> Encoder {
    let NestedLeafCtx {
        access: _,
        idx,
        parent_name,
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let pp = crate::codegen::polars_paths::prelude();
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let positions = &ids.positions;
    let df = &ids.df;
    let take = &ids.take;

    let direct_inner = quote! {
        #df.column(__df_derive_col_name)?
            .as_materialized_series()
            .clone()
    };
    let take_inner = quote! {{
        let __df_derive_inner_full = #df
            .column(__df_derive_col_name)?
            .as_materialized_series();
        __df_derive_inner_full.take(&#take)?
    }};
    let null_inner = quote! {
        #pp::Series::new_empty("".into(), __df_derive_dtype)
            .extend_constant(#pp::AnyValue::Null, items.len())?
    };

    let scan = quote! {
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::with_capacity(items.len());
        let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for __df_derive_it in items {
            match #match_expr {
                ::std::option::Option::Some(__df_derive_v) => {
                    #positions.push(::std::option::Option::Some(
                        #flat.len() as #pp::IdxSize,
                    ));
                    #flat.push(__df_derive_v);
                }
                ::std::option::Option::None => {
                    #positions.push(::std::option::Option::None);
                }
            }
        }
    };
    let consume_direct = nested_consume_columns(parent_name, to_df_trait, ty, &direct_inner);
    let consume_take = nested_consume_columns(parent_name, to_df_trait, ty, &take_inner);
    let consume_null = nested_consume_columns(parent_name, to_df_trait, ty, &null_inner);
    let columnar_block = quote! {{
        #scan
        if #flat.is_empty() {
            #consume_null
        } else if #flat.len() == items.len() {
            let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
            #consume_direct
        } else {
            let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
            let #take: #pp::IdxCa =
                <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #positions.iter().copied(),
                );
            #consume_take
        }
    }};
    Encoder::Multi {
        columnar: columnar_block,
    }
}

// --- Generalized depth-N nested encoder ---

/// Per-layer ident set for the depth-N nested encoder. Layer 0 is the
/// outermost `Vec`. `offsets` accumulates offset entries (one per child-list
/// inside this layer); `offsets_buf` is the frozen `OffsetsBuffer`. `validity`
/// holds the layer-level `MutableBitmap` when this layer's outer Option is
/// present. `bind` is the per-layer iteration binding (mirrors the primitive
/// `VecLayerIdents` pattern).
struct NestedLayerIdents {
    offsets: syn::Ident,
    offsets_buf: syn::Ident,
    validity_mb: syn::Ident,
    validity_bm: syn::Ident,
    bind: syn::Ident,
}

fn nested_layer_idents(idx: usize, depth: usize) -> Vec<NestedLayerIdents> {
    (0..depth)
        .map(|i| NestedLayerIdents {
            offsets: format_ident!("__df_derive_n_off_{}_{}", idx, i),
            offsets_buf: format_ident!("__df_derive_n_off_buf_{}_{}", idx, i),
            validity_mb: format_ident!("__df_derive_n_valmb_{}_{}", idx, i),
            validity_bm: format_ident!("__df_derive_n_valbm_{}_{}", idx, i),
            bind: format_ident!("__df_derive_n_bind_{}_{}", idx, i),
        })
        .collect()
}

/// Build the depth-N nested vec encoder for an arbitrary [`VecShape`].
/// Handles per-layer outer-Option (validity bitmap), inner-Option
/// (per-element positions + scatter via `IdxCa::take`), and any mix
/// thereof. Replaces the seven hand-written shape variants.
#[allow(clippy::too_many_lines)]
fn nested_vec_encoder_general(ctx: &NestedLeafCtx<'_>, shape: &VecShape) -> Encoder {
    let NestedLeafCtx {
        access,
        idx,
        parent_name,
        ty,
        columnar_trait,
        to_df_trait,
        pa_root,
    } = *ctx;
    let pp = crate::codegen::polars_paths::prelude();
    let depth = shape.depth();
    let layers = nested_layer_idents(idx, depth);
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let positions = &ids.positions;
    let df = &ids.df;
    let take = &ids.take;
    let total = &ids.total;

    let scan_body = build_nested_scan_body(access, shape, &layers, flat, positions, total, ty);
    let validity_freeze = build_nested_validity_freeze(shape, &layers, pa_root);
    let offsets_freeze = build_nested_offsets_freeze(&layers, pa_root);

    // Per-column wrap expressions. We need three branches:
    // - filled-direct: `df` exists, positions match flat 1:1 (or no inner
    //   option) → wrap each `df.column(name)` chunk in N list layers.
    // - filled-take: `df` exists, positions has Nones → take with IdxCa,
    //   wrap result in N list layers.
    // - all-empty (flat empty): wrap an empty Series in N list layers.

    let inner_col_direct = quote! {
        #df.column(__df_derive_col_name)?
            .as_materialized_series()
            .clone()
    };
    let inner_col_take = quote! {{
        let __df_derive_inner_full = #df
            .column(__df_derive_col_name)?
            .as_materialized_series();
        __df_derive_inner_full.take(&#take)?
    }};
    let inner_col_empty = quote! {
        #pp::Series::new_empty("".into(), __df_derive_dtype)
    };
    // All-absent: every element slot is `None`, but the outer offsets are
    // non-zero (each outer row carries inner-Vec lengths > 0). The inner
    // chunk must be a typed-null Series of length `total` so the offsets
    // buffer's max value (which equals `total`) doesn't exceed the chunk's
    // length. Without inner-Option, this branch is unreachable — zero
    // total leaves implies zero outer-list members.
    let inner_col_all_absent = quote! {
        #pp::Series::new_empty("".into(), __df_derive_dtype)
            .extend_constant(#pp::AnyValue::Null, #total)?
    };

    let series_direct = build_nested_layer_wrap(&layers, shape, &inner_col_direct, &pp);
    let series_take = build_nested_layer_wrap(&layers, shape, &inner_col_take, &pp);
    let series_empty = build_nested_layer_wrap(&layers, shape, &inner_col_empty, &pp);
    let series_all_absent = build_nested_layer_wrap(&layers, shape, &inner_col_all_absent, &pp);

    let consume_direct = nested_consume_columns(parent_name, to_df_trait, ty, &series_direct);
    let consume_take = nested_consume_columns(parent_name, to_df_trait, ty, &series_take);
    let consume_empty = nested_consume_columns(parent_name, to_df_trait, ty, &series_empty);
    let consume_all_absent =
        nested_consume_columns(parent_name, to_df_trait, ty, &series_all_absent);
    let columnar_block = if shape.has_inner_option() {
        // 4-branch dispatch for the inner-Option case:
        // - total == 0: no leaf slots at all (every outer Vec was empty
        //   or every outer Option was None) → empty inner Series.
        // - flat.is_empty() (but total > 0): every leaf slot was None →
        //   typed-null Series of length total, offsets reference it.
        // - flat.len() == total: every leaf slot was Some → direct.
        // - else: mixed → take.
        //
        // The offsets-buffer freeze is emitted INSIDE each arm rather than
        // hoisted above the dispatch: with the freeze local to each branch,
        // LLVM specializes register allocation around the heavily-inlined
        // `columnar_from_refs` on the hot direct/take paths instead of
        // treating the `OffsetsBuffer` construction as a global obligation.
        quote! {{
            #scan_body
            #validity_freeze
            if #total == 0 {
                #offsets_freeze
                #consume_empty
            } else if #flat.is_empty() {
                #offsets_freeze
                #consume_all_absent
            } else if #flat.len() == #total {
                let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
                #offsets_freeze
                #consume_direct
            } else {
                let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
                let #take: #pp::IdxCa =
                    <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                        "".into(),
                        #positions.iter().copied(),
                    );
                #offsets_freeze
                #consume_take
            }
        }}
    } else {
        // No inner-Option: total == flat.len(). Two branches: empty
        // when flat is empty (no leaves), direct otherwise. Same
        // freeze-inside-branch rationale as above.
        quote! {{
            #scan_body
            #validity_freeze
            if #flat.is_empty() {
                #offsets_freeze
                #consume_empty
            } else {
                let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
                #offsets_freeze
                #consume_direct
            }
        }}
    };
    Encoder::Multi {
        columnar: columnar_block,
    }
}

/// Scan the input building flat refs, per-layer offsets vecs, per-layer
/// validity bitmaps, and (when `has_inner_option`) per-element positions.
///
/// The scan walker (per-row offset/validity/leaf-push body) is shared with
/// the flat-vec path via [`ShapeScan`]; the precount stays inline because
/// its dead-store-after-precount counter behavior differs from any reuse
/// the flat-vec path needs (and is bench-sensitive).
#[allow(clippy::items_after_statements, clippy::too_many_lines)]
fn build_nested_scan_body(
    access: &TokenStream,
    shape: &VecShape,
    layers: &[NestedLayerIdents],
    flat: &syn::Ident,
    positions: &syn::Ident,
    total: &syn::Ident,
    ty: &TokenStream,
) -> TokenStream {
    let pp = crate::codegen::polars_paths::prelude();
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    let depth = shape.depth();

    // Per-layer counter for sizing offsets vecs and validity bitmaps.
    // `total_layer_{i}` counts the number of child-lists inside layer `i`
    // (i.e. how many entries layer `i+1` will accumulate). Layer 0 is sized
    // by `items.len()` directly. `total_leaves` (== `total`) is used for the
    // flat ref vec capacity (and the positions vec under has_inner_option).
    let layer_counters: Vec<syn::Ident> = (0..depth.saturating_sub(1))
        .map(|i| format_ident!("__df_derive_n_total_layer_{}", i))
        .collect();

    // Deepest-layer leaf body: scatter into flat (and positions, under
    // inner-Option). The nested-struct path normalizes nested-struct wrappers
    // so `inner_option_layers` is at most 1 — assert that here so the shared
    // walker's multi-collapse branch can't accidentally activate via a future
    // emitter change.
    debug_assert!(
        shape.inner_option_layers <= 1,
        "nested-struct scan walker only supports inner_option_layers <= 1"
    );
    let leaf_body = |vec_bind: &TokenStream| -> TokenStream {
        if shape.has_inner_option() {
            quote! {
                for __df_derive_maybe in #vec_bind.iter() {
                    match __df_derive_maybe {
                        ::std::option::Option::Some(__df_derive_v) => {
                            #positions.push(::std::option::Option::Some(
                                #flat.len() as #pp::IdxSize,
                            ));
                            #flat.push(__df_derive_v);
                        }
                        ::std::option::Option::None => {
                            #positions.push(::std::option::Option::None);
                        }
                    }
                }
            }
        } else {
            quote! {
                for __df_derive_v in #vec_bind.iter() {
                    #flat.push(__df_derive_v);
                }
            }
        }
    };
    let leaf_offsets_post_push = if shape.has_inner_option() {
        quote! { #positions.len() }
    } else {
        quote! { #flat.len() }
    };
    let scan_layers: Vec<ScanLayerIdents<'_>> = layers
        .iter()
        .map(|l| ScanLayerIdents {
            offsets: &l.offsets,
            validity: &l.validity_mb,
            bind: &l.bind,
        })
        .collect();
    let scan_iter = ShapeScan {
        shape,
        access,
        layers: &scan_layers,
        outer_some_prefix: "__df_derive_n_some_",
        leaf_body: &leaf_body,
        leaf_offsets_post_push: &leaf_offsets_post_push,
    }
    .build();

    // Precount: walk the same structure and tally totals.
    //
    // The scan loop must NOT increment the per-layer counters that the
    // precount loop sized — those counters are dead-store after the precount
    // and re-incrementing them during scan only inflates loop bodies (LLVM
    // does eliminate the dead store, but the path is sensitive to exactly
    // how the loop is shaped). Counters live exclusively in `build_pre_iter`.
    fn build_pre_iter(
        shape: &VecShape,
        layers: &[NestedLayerIdents],
        layer_counters: &[syn::Ident],
        total: &syn::Ident,
        cur: usize,
        vec_bind: &TokenStream,
    ) -> TokenStream {
        let depth = shape.depth();
        if cur + 1 == depth {
            quote! { #total += #vec_bind.len(); }
        } else {
            let inner_bind = &layers[cur + 1].bind;
            let counter = &layer_counters[cur];
            let inner_pre = build_pre_layer(
                shape,
                layers,
                layer_counters,
                total,
                cur + 1,
                &quote! { #inner_bind },
            );
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_pre
                    #counter += 1;
                }
            }
        }
    }

    fn build_pre_layer(
        shape: &VecShape,
        layers: &[NestedLayerIdents],
        layer_counters: &[syn::Ident],
        total: &syn::Ident,
        cur: usize,
        bind: &TokenStream,
    ) -> TokenStream {
        if shape.layers[cur].has_outer_validity() {
            let inner_vec_bind = format_ident!("__df_derive_n_pre_some_{}", cur);
            let inner = build_pre_iter(
                shape,
                layers,
                layer_counters,
                total,
                cur,
                &quote! { #inner_vec_bind },
            );
            quote! {
                if let ::std::option::Option::Some(#inner_vec_bind) = #bind {
                    #inner
                }
            }
        } else {
            build_pre_iter(shape, layers, layer_counters, total, cur, bind)
        }
    }

    let layer0_iter_src = quote! { (&(#access)) };
    let pre_iter_body = build_pre_layer(shape, layers, &layer_counters, total, 0, &layer0_iter_src);

    // Allocate offsets vecs and validity bitmaps via the shared helpers.
    // The per-depth counter ident matches the precount loop's counters above.
    let counter_for_depth = |i: usize| -> TokenStream {
        let id = &layer_counters[i];
        quote! { #id }
    };
    let offsets_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.validity_mb).collect();
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls =
        shape_validity_decls(shape, &validity_idents, &counter_for_depth, &pa_root);

    let counter_decls = layer_counters
        .iter()
        .map(|c| quote! { let mut #c: usize = 0; });
    let positions_decl = if shape.has_inner_option() {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total);
        }
    } else {
        TokenStream::new()
    };
    quote! {
        let mut #total: usize = 0;
        #(#counter_decls)*
        for __df_derive_it in items {
            #pre_iter_body
        }
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::with_capacity(#total);
        #positions_decl
        #offsets_decls
        #validity_decls
        #scan_iter
    }
}

/// Freeze the per-layer `MutableBitmap` validity buffers into immutable
/// `Bitmap`s (only for layers whose Option is present). Each layer's bitmap
/// is named `validity_bm_<idx>_<layer>` post-freeze.
fn build_nested_validity_freeze(
    shape: &VecShape,
    layers: &[NestedLayerIdents],
    pa_root: &TokenStream,
) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if !shape.layers[i].has_outer_validity() {
            continue;
        }
        let mb = &layer.validity_mb;
        let bm = &layer.validity_bm;
        out.push(quote! {
            let #bm: #pa_root::bitmap::Bitmap =
                <#pa_root::bitmap::Bitmap as ::core::convert::From<
                    #pa_root::bitmap::MutableBitmap,
                >>::from(#mb);
        });
    }
    quote! { #(#out)* }
}

/// Freeze each layer's offsets vec into an `OffsetsBuffer`. Layer 0 is
/// always populated. When all layers are empty (flat is empty), the
/// offsets vecs still contain at least the leading 0; the freeze still
/// succeeds because `OffsetsBuffer::try_from(vec![0])` is valid.
///
/// The freeze consumes each offsets `Vec<i64>` into the buffer (no clone).
/// `OffsetsBuffer` is `Arc`-backed, so subsequent uses (one per branch in
/// the four-way dispatch, plus per-layer wraps) clone cheaply by bumping
/// the refcount.
fn build_nested_offsets_freeze(layers: &[NestedLayerIdents], pa_root: &TokenStream) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::new();
    for layer in layers {
        let offsets = &layer.offsets;
        let buf = &layer.offsets_buf;
        out.push(quote! {
            let #buf: #pa_root::offset::OffsetsBuffer<i64> =
                #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
        });
    }
    quote! { #(#out)* }
}

/// Wrap the supplied inner-column expression in `depth` `LargeListArray::new`
/// layers (innermost-first, outermost-last) and route the outermost through
/// `__df_derive_assemble_list_series_unchecked`. Per-layer validity bitmaps
/// (when their Option is present) ride under each `LargeListArray`.
fn build_nested_layer_wrap(
    layers: &[NestedLayerIdents],
    shape: &VecShape,
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let depth = layers.len();
    let mut block: Vec<TokenStream> = Vec::new();
    block.push(quote! {
        let __df_derive_inner_col: #pp::Series = #inner_col_expr;
        let __df_derive_inner_rech = __df_derive_inner_col.rechunk();
        let __df_derive_inner_chunk: #pp::ArrayRef =
            __df_derive_inner_rech.chunks()[0].clone();
    });
    let mut prev_arr = format_ident!("__df_derive_inner_chunk");
    for cur in (0..depth).rev() {
        let layer = &layers[cur];
        let buf = &layer.offsets_buf;
        let arr_id = format_ident!("__df_derive_n_arr_{}", cur);
        let validity_expr = if shape.layers[cur].has_outer_validity() {
            let bm = &layer.validity_bm;
            quote! { ::std::option::Option::Some(::std::clone::Clone::clone(&#bm)) }
        } else {
            quote! { ::std::option::Option::None }
        };
        let prev = prev_arr.clone();
        // The first wrap consumes the inner chunk (an `ArrayRef`); subsequent
        // wraps consume the previous LargeListArray boxed as ArrayRef.
        let prev_payload = if cur == depth - 1 {
            quote! { #prev }
        } else {
            quote! { ::std::boxed::Box::new(#prev) as #pp::ArrayRef }
        };
        let pa_root = crate::codegen::polars_paths::polars_arrow_root();
        // Read the chunk's arrow dtype: `ArrayRef`'s dtype method (the
        // first wrap) and `LargeListArray::dtype()` (subsequent wraps)
        // both proxy through the `Array` trait.
        let dtype_src = if cur == depth - 1 {
            quote! { #prev.dtype().clone() }
        } else {
            quote! { #pa_root::array::Array::dtype(&#prev).clone() }
        };
        block.push(quote! {
            let #arr_id: #pp::LargeListArray = #pp::LargeListArray::new(
                #pp::LargeListArray::default_datatype(#dtype_src),
                ::std::clone::Clone::clone(&#buf),
                #prev_payload,
                #validity_expr,
            );
        });
        prev_arr = arr_id;
    }
    // Wrap the per-leaf logical dtype in `(depth - 1)` extra `List<>` layers
    // to construct what `__df_derive_assemble_list_series_unchecked` expects
    // (the helper wraps once more, yielding the full N-layer List nesting).
    let mut helper_logical = quote! { (*__df_derive_dtype).clone() };
    for _ in 0..depth.saturating_sub(1) {
        helper_logical = quote! { #pp::DataType::List(::std::boxed::Box::new(#helper_logical)) };
    }
    let outer = prev_arr;
    quote! {{
        #(#block)*
        __df_derive_assemble_list_series_unchecked(
            #outer,
            #helper_logical,
        )
    }}
}

/// Top-level dispatcher for the nested-struct/generic encoder paths.
/// After Step 4 this covers every wrapper stack the parser accepts —
/// the `[]` and `[Option]` shapes use dedicated leaf encoders; every
/// `Vec`-bearing shape (including deep nestings, mid-stack `Option`s,
/// outer-list validity) routes through the depth-N general encoder.
pub fn build_nested_encoder(wrappers: &[Wrapper], ctx: &NestedLeafCtx<'_>) -> Encoder {
    match normalize_wrappers(wrappers) {
        WrapperKind::Leaf { option_layers: 0 } => nested_leaf_encoder(ctx),
        WrapperKind::Leaf {
            option_layers: layers,
        } => {
            // Collapse N consecutive Options into a single `Option<&T>`
            // before invoking the option-leaf encoder. Polars folds every
            // nested None into one validity bit, so `Some(None)` and
            // outer `None` produce the same `AnyValue::Null`. The
            // intermediate access expression is `(...).as_ref().and_then(...)`
            // which evaluates to `Option<&T>` and matches the option-leaf
            // encoder's expected access type for the single-Option case.
            let collapsed_access = if layers >= 2 {
                let chain = collapse_options_to_ref(ctx.access, layers);
                quote! { (#chain) }
            } else {
                ctx.access.clone()
            };
            let new_ctx = NestedLeafCtx {
                access: &collapsed_access,
                idx: ctx.idx,
                parent_name: ctx.parent_name,
                ty: ctx.ty,
                columnar_trait: ctx.columnar_trait,
                to_df_trait: ctx.to_df_trait,
                pa_root: ctx.pa_root,
            };
            nested_option_encoder_collapsed(&new_ctx, layers)
        }
        WrapperKind::Vec(shape) => nested_vec_encoder_general(ctx, &shape),
    }
}
