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
use quote::quote;

use super::idents;
use super::shape_walk::{
    LayerIdents, LayerWrap, OwnPolicy, ShapePrecount, ShapeScan, shape_assemble_list_stack,
    shape_offsets_decls, shape_validity_decls,
};
use super::{BaseCtx, Encoder, VecShape, WrapperKind, collapse_options_to_ref, normalize_wrappers};

/// Per-call-site context for nested-struct/generic encoders. Carries the
/// `polars-arrow` crate root (so the combinators don't re-resolve it per
/// call) plus the type-as-path expression and the fully-qualified trait
/// paths used in UFCS calls (`<#ty as #columnar_trait>::columnar_from_refs`,
/// `<#ty as #to_df_trait>::schema`).
pub struct NestedLeafCtx<'a> {
    pub base: BaseCtx<'a>,
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
            flat: idents::nested_flat(idx),
            positions: idents::nested_positions(idx),
            df: idents::nested_df(idx),
            take: idents::nested_take(idx),
            total: idents::nested_total(idx),
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
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let prefixed = idents::nested_prefixed_name();
    let inner = idents::nested_inner_series();
    let named = idents::field_named_series();
    quote! {
        for (#col_name, #dtype) in
            <#ty as #to_df_trait>::schema()?
        {
            let #col_name: &str = #col_name.as_str();
            let #dtype: &#pp::DataType = &#dtype;
            {
                let #prefixed = ::std::format!(
                    "{}.{}", #parent_name, #col_name,
                );
                let #inner: #pp::Series = #series_expr;
                let #named = #inner
                    .with_name(#prefixed.as_str().into());
                columns.push(#named.into());
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
        base: BaseCtx { access, idx, name },
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let df = &ids.df;
    let col_name = idents::nested_col_name();
    let inner_expr = quote! {
        #df.column(#col_name)?
            .as_materialized_series()
            .clone()
    };
    let columnar = nested_consume_columns(name, to_df_trait, ty, &inner_expr);
    let it = idents::populator_iter();
    let setup = quote! {
        let #flat: ::std::vec::Vec<&#ty> = items
            .iter()
            .map(|#it| &(#access))
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
    let access_ts = ctx.base.access.clone();
    let match_expr = if option_layers >= 2 {
        quote! { (#access_ts) }
    } else {
        quote! { &(#access_ts) }
    };
    nested_option_encoder_impl(ctx, &match_expr)
}

fn nested_option_encoder_impl(ctx: &NestedLeafCtx<'_>, match_expr: &TokenStream) -> Encoder {
    let NestedLeafCtx {
        base: BaseCtx {
            access: _,
            idx,
            name,
        },
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
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();

    let direct_inner = quote! {
        #df.column(#col_name)?
            .as_materialized_series()
            .clone()
    };
    let take_inner = quote! {{
        let #inner_full = #df
            .column(#col_name)?
            .as_materialized_series();
        #inner_full.take(&#take)?
    }};
    let null_inner = quote! {
        #pp::Series::new_empty("".into(), #dtype)
            .extend_constant(#pp::AnyValue::Null, items.len())?
    };

    let it = idents::populator_iter();
    let v = idents::leaf_value();
    let scan = quote! {
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::with_capacity(items.len());
        let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for #it in items {
            match #match_expr {
                ::std::option::Option::Some(#v) => {
                    #positions.push(::std::option::Option::Some(
                        #flat.len() as #pp::IdxSize,
                    ));
                    #flat.push(#v);
                }
                ::std::option::Option::None => {
                    #positions.push(::std::option::Option::None);
                }
            }
        }
    };
    let consume_direct = nested_consume_columns(name, to_df_trait, ty, &direct_inner);
    let consume_take = nested_consume_columns(name, to_df_trait, ty, &take_inner);
    let consume_null = nested_consume_columns(name, to_df_trait, ty, &null_inner);
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

/// Per-(field, layer) ident set for the depth-N nested encoder. Layer 0 is
/// the outermost `Vec`. `offsets` accumulates offset entries (one per
/// child-list inside this layer); `offsets_buf` is the frozen `OffsetsBuffer`.
/// `validity_mb` holds the layer-level `MutableBitmap` when this layer's
/// outer Option is present (frozen into `validity_bm` after the scan).
fn nested_layer_idents(idx: usize, depth: usize) -> Vec<LayerIdents> {
    (0..depth)
        .map(|i| LayerIdents {
            offsets: idents::nested_layer_offsets(idx, i),
            offsets_buf: idents::nested_layer_offsets_buf(idx, i),
            validity_mb: idents::nested_layer_validity_mb(idx, i),
            validity_bm: idents::nested_layer_validity_bm(idx, i),
            bind: idents::nested_layer_bind(idx, i),
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
        base: BaseCtx { access, idx, name },
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
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();

    // Per-column wrap expressions. We need three branches:
    // - filled-direct: `df` exists, positions match flat 1:1 (or no inner
    //   option) → wrap each `df.column(name)` chunk in N list layers.
    // - filled-take: `df` exists, positions has Nones → take with IdxCa,
    //   wrap result in N list layers.
    // - all-empty (flat empty): wrap an empty Series in N list layers.

    let inner_col_direct = quote! {
        #df.column(#col_name)?
            .as_materialized_series()
            .clone()
    };
    let inner_col_take = quote! {{
        let #inner_full = #df
            .column(#col_name)?
            .as_materialized_series();
        #inner_full.take(&#take)?
    }};
    let inner_col_empty = quote! {
        #pp::Series::new_empty("".into(), #dtype)
    };
    // All-absent: every element slot is `None`, but the outer offsets are
    // non-zero (each outer row carries inner-Vec lengths > 0). The inner
    // chunk must be a typed-null Series of length `total` so the offsets
    // buffer's max value (which equals `total`) doesn't exceed the chunk's
    // length. Without inner-Option, this branch is unreachable — zero
    // total leaves implies zero outer-list members.
    let inner_col_all_absent = quote! {
        #pp::Series::new_empty("".into(), #dtype)
            .extend_constant(#pp::AnyValue::Null, #total)?
    };

    let series_direct = build_nested_layer_wrap(&layers, shape, &inner_col_direct, &pp);
    let series_take = build_nested_layer_wrap(&layers, shape, &inner_col_take, &pp);
    let series_empty = build_nested_layer_wrap(&layers, shape, &inner_col_empty, &pp);
    let series_all_absent = build_nested_layer_wrap(&layers, shape, &inner_col_all_absent, &pp);

    let consume_direct = nested_consume_columns(name, to_df_trait, ty, &series_direct);
    let consume_take = nested_consume_columns(name, to_df_trait, ty, &series_take);
    let consume_empty = nested_consume_columns(name, to_df_trait, ty, &series_empty);
    let consume_all_absent = nested_consume_columns(name, to_df_trait, ty, &series_all_absent);
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
/// Both the scan walker (per-row offset/validity/leaf-push body) and the
/// precount walker (per-layer-counter tally) are shared with the flat-vec
/// path via [`ShapeScan`] and [`ShapePrecount`]. The scan loop must NOT
/// increment the per-layer counters that the precount loop sized — those
/// counters are dead-store after the precount and re-incrementing them
/// during scan only inflates loop bodies (LLVM does eliminate the dead
/// store, but the path is sensitive to exactly how the loop is shaped).
fn build_nested_scan_body(
    access: &TokenStream,
    shape: &VecShape,
    layers: &[LayerIdents],
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
        .map(idents::nested_layer_total)
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
    let maybe = idents::nested_maybe();
    let v = idents::leaf_value();
    let leaf_body = |vec_bind: &TokenStream| -> TokenStream {
        if shape.has_inner_option() {
            quote! {
                for #maybe in #vec_bind.iter() {
                    match #maybe {
                        ::std::option::Option::Some(#v) => {
                            #positions.push(::std::option::Option::Some(
                                #flat.len() as #pp::IdxSize,
                            ));
                            #flat.push(#v);
                        }
                        ::std::option::Option::None => {
                            #positions.push(::std::option::Option::None);
                        }
                    }
                }
            }
        } else {
            quote! {
                for #v in #vec_bind.iter() {
                    #flat.push(#v);
                }
            }
        }
    };
    let leaf_offsets_post_push = if shape.has_inner_option() {
        quote! { #positions.len() }
    } else {
        quote! { #flat.len() }
    };
    let scan_iter = ShapeScan {
        shape,
        access,
        layers,
        outer_some_prefix: idents::NESTED_OUTER_SOME_PREFIX,
        leaf_body: &leaf_body,
        leaf_offsets_post_push: &leaf_offsets_post_push,
    }
    .build();

    // The nested precount uses a distinct outer-Some prefix
    // (`__df_derive_n_pre_some_`) from the scan walker
    // (`__df_derive_n_some_`) so the two loops can coexist without name
    // shadowing inside the same generated block. Both prefixes live in
    // [`idents`] so any rename touches one place.
    let pre_iter_body = ShapePrecount {
        shape,
        access,
        layers,
        outer_some_prefix: idents::NESTED_PRE_OUTER_SOME_PREFIX,
        total_counter: total,
        layer_counters: &layer_counters,
    }
    .build();

    // Allocate offsets vecs and validity bitmaps via the shared helpers.
    // The per-depth counter ident matches the precount loop's counters above.
    let counter_for_depth = |i: usize| idents::nested_layer_total_token(i);
    let offsets_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.validity_mb).collect();
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls =
        shape_validity_decls(shape, &validity_idents, &counter_for_depth, &pa_root);

    let positions_decl = if shape.has_inner_option() {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total);
        }
    } else {
        TokenStream::new()
    };
    quote! {
        #pre_iter_body
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
    layers: &[LayerIdents],
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
fn build_nested_offsets_freeze(layers: &[LayerIdents], pa_root: &TokenStream) -> TokenStream {
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
/// `__df_derive_assemble_list_series_unchecked` via the shared
/// [`shape_assemble_list_stack`] helper. Per-layer validity bitmaps (when
/// their Option is present) ride under each `LargeListArray`. The freeze
/// of offsets/validity already happened above the four-arm dispatch so this
/// helper just wires the pre-frozen idents into the shared stack.
fn build_nested_layer_wrap(
    layers: &[LayerIdents],
    shape: &VecShape,
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let inner_chunk = idents::nested_inner_chunk();
    let inner_col = idents::nested_inner_col();
    let inner_rech = idents::nested_inner_rech();
    let chunk_decl = quote! {
        let #inner_col: #pp::Series = #inner_col_expr;
        let #inner_rech = #inner_col.rechunk();
        let #inner_chunk: #pp::ArrayRef =
            #inner_rech.chunks()[0].clone();
    };
    let wrap_layers: Vec<LayerWrap<'_>> = layers
        .iter()
        .enumerate()
        .map(|(i, layer)| LayerWrap {
            // The frozen offsets buffer is bound above the four-arm
            // dispatch's `for col in schema { ... }` iteration body — it
            // is read on every iteration of every arm, so the helper
            // must clone it (an `Arc::clone` under the hood). Moving
            // would fail the borrow check on the second iteration.
            offsets_buf: OwnPolicy::Clone(&layer.offsets_buf),
            validity_bm: shape.layers[i]
                .has_outer_validity()
                .then_some(&layer.validity_bm),
            // Nested freezes happen once above the four-arm dispatch (see
            // `build_nested_offsets_freeze` / `build_nested_validity_freeze`)
            // so the per-layer wrap doesn't re-freeze.
            freeze_decl: TokenStream::new(),
        })
        .collect();
    // The inner chunk is already an `ArrayRef` (`Box<dyn Array>`); the
    // dtype access goes through the trait object's vtable (no static-
    // dispatch alternative exists for a chunk that came out of a
    // `Series` rechunk). The dtype borrow is taken before the chunk is
    // moved into the innermost wrap below.
    let seed_dtype = quote! { #inner_chunk.dtype().clone() };
    let dtype = idents::nested_col_dtype();
    let stack = shape_assemble_list_stack(
        quote! { #inner_chunk },
        seed_dtype,
        &wrap_layers,
        quote! { (*#dtype).clone() },
        &idents::nested_layer_list_arr,
    );
    quote! {{
        #chunk_decl
        #stack
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
                let chain = collapse_options_to_ref(ctx.base.access, layers);
                quote! { (#chain) }
            } else {
                ctx.base.access.clone()
            };
            let new_ctx = NestedLeafCtx {
                base: BaseCtx {
                    access: &collapsed_access,
                    idx: ctx.base.idx,
                    name: ctx.base.name,
                },
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
