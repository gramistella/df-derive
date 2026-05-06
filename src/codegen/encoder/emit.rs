//! Unified depth-N `Vec`-bearing emitter parameterized by [`LeafKind`].
//!
//! [`vec_emit_general`] is the single place that ties together the
//! depth-N walker primitives ([`ShapePrecount`], [`ShapeScan`],
//! [`shape_offsets_decls`], [`shape_validity_decls`],
//! [`shape_assemble_list_stack`]). It dispatches on [`LeafKind`] for the
//! points where per-element-push and collect-then-bulk genuinely diverge.
//!
//! See `docs/encoder-ir.md` for the conceptual model.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::VecLayers;

use super::collapse_options_to_ref;
use super::idents;
use super::leaf_kind::{CollectThenBulk, LeafKind};
use super::shape_walk::{
    LayerIdents, LayerWrap, ShapePrecount, ShapeScan, shape_assemble_list_stack,
    shape_offsets_decls, shape_validity_decls,
};

/// Per-layer ident bundle factory. `field_idx == None` produces the
/// per-element-push path's flat-vec idents (`__df_derive_layer_*_{layer}`,
/// no field namespacing — the emit block is scoped per field via
/// `__df_derive_field_series_<idx>`); `field_idx == Some(idx)` produces the
/// collect-then-bulk path's per-(field, layer) namespaced idents
/// (`__df_derive_n_*_{idx}_{layer}`).
fn layer_idents(field_idx: Option<usize>, layer_idx: usize) -> LayerIdents {
    field_idx.map_or_else(
        || LayerIdents {
            offsets: idents::vec_layer_offsets(layer_idx),
            offsets_buf: idents::vec_layer_offsets_buf(layer_idx),
            validity_mb: idents::vec_layer_validity(layer_idx),
            validity_bm: idents::vec_layer_validity_bm(layer_idx),
            bind: idents::vec_layer_bind(layer_idx),
        },
        |idx| LayerIdents {
            offsets: idents::nested_layer_offsets(idx, layer_idx),
            offsets_buf: idents::nested_layer_offsets_buf(idx, layer_idx),
            validity_mb: idents::nested_layer_validity_mb(idx, layer_idx),
            validity_bm: idents::nested_layer_validity_bm(idx, layer_idx),
            bind: idents::nested_layer_bind(idx, layer_idx),
        },
    )
}

/// Build the per-layer `LayerWrap` slice the shared list-stack helper
/// consumes. Each layer's `freeze_decl` is empty for the
/// collect-then-bulk path (freezes hoisted above) and contains the
/// `OffsetsBuffer::try_from(...)?` plus optional `Bitmap::from(...)`
/// for the per-element-push path (freezes interleaved with each wrap).
fn layer_wraps<'a>(
    shape: &VecLayers,
    layers: &'a [LayerIdents],
    kind: &LeafKind<'_>,
    pa_root: &TokenStream,
) -> Vec<LayerWrap<'a>> {
    let mut out: Vec<LayerWrap<'_>> = Vec::with_capacity(shape.depth());
    for (cur, layer) in layers.iter().enumerate() {
        let buf_id = &layer.offsets_buf;
        let validity_bm = if shape.layers[cur].has_outer_validity() {
            Some(&layer.validity_bm)
        } else {
            None
        };
        let freeze_decl = if kind.freeze_hoisted() {
            TokenStream::new()
        } else {
            let offsets = &layer.offsets;
            let mut fd = quote! {
                let #buf_id: #pa_root::offset::OffsetsBuffer<i64> =
                    #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
            };
            if let Some(bm_id) = validity_bm {
                let validity_mb = &layer.validity_mb;
                fd.extend(quote! {
                    let #bm_id: #pa_root::bitmap::Bitmap =
                        <#pa_root::bitmap::Bitmap as ::core::convert::From<
                            #pa_root::bitmap::MutableBitmap,
                        >>::from(#validity_mb);
                });
            }
            fd
        };
        out.push(LayerWrap {
            offsets_buf: kind.layer_own_policy(buf_id),
            validity_bm,
            freeze_decl,
        });
    }
    out
}

/// Build the hoisted-freeze pair for the collect-then-bulk path:
/// converts each layer's `MutableBitmap` to `Bitmap` (where the layer
/// has an outer Option) and each layer's `Vec<i64>` to
/// `OffsetsBuffer<i64>`. Returns empty token streams for the
/// per-element-push path (freezes interleaved per-layer instead).
///
/// The pair is `(validity_freeze, offsets_freeze)` — the call site
/// emits validity first (any outer-Option arms reference the frozen
/// `Bitmap` lifetime) then offsets at the head of each branch.
fn hoisted_freezes(
    shape: &VecLayers,
    layers: &[LayerIdents],
    kind: &LeafKind<'_>,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream) {
    if !kind.freeze_hoisted() {
        return (TokenStream::new(), TokenStream::new());
    }
    let mut validity_freeze: Vec<TokenStream> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if !shape.layers[i].has_outer_validity() {
            continue;
        }
        let mb = &layer.validity_mb;
        let bm = &layer.validity_bm;
        validity_freeze.push(quote! {
            let #bm: #pa_root::bitmap::Bitmap =
                <#pa_root::bitmap::Bitmap as ::core::convert::From<
                    #pa_root::bitmap::MutableBitmap,
                >>::from(#mb);
        });
    }
    let mut offsets_freeze: Vec<TokenStream> = Vec::new();
    for layer in layers {
        let offsets = &layer.offsets;
        let buf = &layer.offsets_buf;
        offsets_freeze.push(quote! {
            let #buf: #pa_root::offset::OffsetsBuffer<i64> =
                #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
        });
    }
    (
        quote! { #(#validity_freeze)* },
        quote! { #(#offsets_freeze)* },
    )
}

/// Build the precount block for a depth-N vec emit. Returns
/// `(precount_decls, total_counter_ident)` — the caller passes the total
/// counter as the leaf-storage-capacity for `Vec::with_capacity` calls.
///
/// `total` and `layer_counters` are owned by the caller (they're idents
/// the helper references in the resulting tokens). The walker is shared
/// with [`super::shape_walk::ShapePrecount`].
fn build_precount<'a>(
    access: &TokenStream,
    shape: &'a VecLayers,
    layers: &'a [LayerIdents],
    outer_some_prefix: &'static str,
    total: &'a syn::Ident,
    layer_counters: &'a [syn::Ident],
) -> TokenStream {
    ShapePrecount {
        shape,
        access,
        layers,
        outer_some_prefix,
        total_counter: total,
        layer_counters,
    }
    .build()
}

/// Build the scan loop for a depth-N vec emit. Drives the deepest-layer
/// for-loop body via `leaf_body`; spliced through [`ShapeScan`].
///
/// `leaf_body` receives the inner-Vec binding (already Option-unwrapped
/// where applicable) and must emit the per-element work (typed-buffer
/// push for the per-element-push path; flat-ref push + optional position
/// scatter for the collect-then-bulk path).
fn build_scan(
    access: &TokenStream,
    shape: &VecLayers,
    layers: &[LayerIdents],
    outer_some_prefix: &'static str,
    leaf_body: &dyn Fn(&TokenStream) -> TokenStream,
    leaf_offsets_post_push: &TokenStream,
) -> TokenStream {
    ShapeScan {
        shape,
        access,
        layers,
        outer_some_prefix,
        leaf_body,
        leaf_offsets_post_push,
    }
    .build()
}

/// Per-element-push leaf body. Wraps the `per_elem_push` token stream in
/// the deepest-layer for-loop, handling the multi-Option collapse for
/// `inner_option_layers > 1` (the bare for-loop binding becomes
/// `__df_derive_v_raw` and is collapsed into `__df_derive_v: Option<&T>`
/// before splicing the push body).
fn pep_leaf_body<'a>(
    shape: &'a VecLayers,
    leaf_bind: &'a syn::Ident,
    per_elem_push: &'a TokenStream,
) -> impl Fn(&TokenStream) -> TokenStream + 'a {
    move |vec_bind: &TokenStream| -> TokenStream {
        if shape.has_inner_option() {
            if shape.inner_option_layers == 1 {
                quote! {
                    for #leaf_bind in #vec_bind.iter() {
                        #per_elem_push
                    }
                }
            } else {
                let raw_bind = idents::leaf_value_raw();
                let collapsed =
                    collapse_options_to_ref(&quote! { #raw_bind }, shape.inner_option_layers);
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind: ::std::option::Option<_> = #collapsed;
                        #per_elem_push
                    }
                }
            }
        } else {
            quote! {
                for #leaf_bind in #vec_bind.iter() {
                    #per_elem_push
                }
            }
        }
    }
}

/// Collect-then-bulk leaf body. The deepest-layer push pushes a `&T` ref
/// into `flat`, plus (when `has_inner_option`) an `Option<IdxSize>` slot
/// into `positions` so the post-scan dispatcher can scatter values back
/// into their original row positions via `IdxCa::take`.
///
/// Debug-asserts `inner_option_layers <= 1` (the parser-injected encoder
/// boundary already enforces this for nested-struct/generic paths; the
/// debug-assert is a safety margin against future emitter changes).
fn ctb_leaf_body<'a>(
    shape: &'a VecLayers,
    flat: &'a syn::Ident,
    positions: &'a syn::Ident,
) -> impl Fn(&TokenStream) -> TokenStream + 'a {
    debug_assert!(
        shape.inner_option_layers <= 1,
        "nested-struct scan walker only supports inner_option_layers <= 1"
    );
    move |vec_bind: &TokenStream| -> TokenStream {
        let pp = crate::codegen::polars_paths::prelude();
        let maybe = idents::nested_maybe();
        let v = idents::leaf_value();
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
    }
}

/// Build the per-column emit body that iterates `<T as ToDataFrame>::schema()?`
/// and pushes each inner-Series-yielding expression onto `columns` with the
/// parent name prefixed. Used by both the bare/option-leaf nested encoders
/// and the collect-then-bulk vec encoder.
pub(super) fn nested_consume_columns(
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

/// Wrap a per-column inner-Series expression in `depth` `LargeListArray::new`
/// layers (innermost-first, outermost-last) and route the outermost through
/// `__df_derive_assemble_list_series_unchecked` via the shared
/// [`shape_assemble_list_stack`] helper. Used by the collect-then-bulk
/// post-scan branches; each branch supplies a different inner-column
/// expression (direct/take/empty/all-absent) but shares the layer-wrap
/// stack.
fn ctb_layer_wrap(
    shape: &VecLayers,
    layers: &[LayerIdents],
    kind: &LeafKind<'_>,
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
    pa_root: &TokenStream,
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
    let wrap_layers = layer_wraps(shape, layers, kind, pa_root);
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

/// Materialize the post-scan tokens for the per-element-push leaf kind:
/// build the typed leaf array, then chain the depth-N `LargeListArray::new`
/// wraps with offsets/validity freezes interleaved per layer (interleaving
/// preserves the historical token shape that benches the depth-N path
/// 4-12% faster than the hoisted alternative — see comment in [`super::vec`]).
fn pep_materialize(
    pep: &super::leaf_kind::PerElementPush,
    shape: &VecLayers,
    layers: &[LayerIdents],
    kind: &LeafKind<'_>,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    // Capture the leaf's arrow dtype to a named local BEFORE boxing the
    // leaf — `Box::new(#leaf_arr) as ArrayRef` moves the typed leaf, so a
    // post-box `Array::dtype(&leaf)` would no longer compile, and a
    // post-box `Array::dtype(&seed)` would dispatch through the boxed trait
    // object's vtable (a virtual call that doesn't inline and reproducibly
    // regresses several depth-N benches by 5-12%).
    let leaf_arr = idents::leaf_arr();
    let seed_arrow_dtype_id = idents::seed_arrow_dtype();
    let seed_dtype_decl = quote! {
        let #seed_arrow_dtype_id: #pa_root::datatypes::ArrowDataType =
            #pa_root::array::Array::dtype(&#leaf_arr).clone();
    };
    let seed = quote! { ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef };
    let seed_dtype = quote! { #seed_arrow_dtype_id };
    let wrap_layers = layer_wraps(shape, layers, kind, pa_root);
    let stack = shape_assemble_list_stack(
        seed,
        seed_dtype,
        &wrap_layers,
        pep.leaf_logical_dtype.clone(),
        &idents::vec_layer_list_arr,
    );
    let leaf_arr_expr = &pep.leaf_arr_expr;
    quote! {
        #leaf_arr_expr
        #seed_dtype_decl
        #stack
    }
}

/// Materialize the post-scan tokens for the collect-then-bulk leaf kind:
/// freeze offsets/validity once above the dispatch (the freezes are read
/// per-arm and per-inner-schema-column, so re-freezing inside each arm
/// would waste work), branch on `(total, flat.len())` to 2 or 4 arms, and
/// per arm iterate the inner schema to wrap each per-column inner Series
/// in N `LargeListArray::new` layers.
///
/// Branch shapes:
/// - `has_inner_option == false`: 2 branches — empty (no leaves) or direct
///   (every leaf is Some, dispatch through `columnar_from_refs` once).
/// - `has_inner_option == true`: 4 branches — `total == 0` (no leaf slots
///   at all), `flat.is_empty() && total > 0` (every leaf slot was None;
///   typed-null Series of length `total`), `flat.len() == total` (every
///   slot was Some; direct), else mixed (build `IdxCa` from `positions`
///   and `take` per inner column).
fn ctb_materialize(
    ctb: &CollectThenBulk<'_>,
    shape: &VecLayers,
    layers: &[LayerIdents],
    kind: &LeafKind<'_>,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let CollectThenBulk {
        ty,
        columnar_trait,
        to_df_trait,
        name,
        idx,
    } = *ctb;
    let flat = idents::nested_flat(idx);
    let positions = idents::nested_positions(idx);
    let df = idents::nested_df(idx);
    let take = idents::nested_take(idx);
    let total = idents::nested_total(idx);
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();

    // Per-column inner-Series expressions for the four branches (or two,
    // when no inner Option). The list-array wrap is identical across
    // branches; only the seed expression differs.
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

    let series_direct = ctb_layer_wrap(shape, layers, kind, &inner_col_direct, pp, pa_root);
    let series_take = ctb_layer_wrap(shape, layers, kind, &inner_col_take, pp, pa_root);
    let series_empty = ctb_layer_wrap(shape, layers, kind, &inner_col_empty, pp, pa_root);
    let series_all_absent =
        ctb_layer_wrap(shape, layers, kind, &inner_col_all_absent, pp, pa_root);

    let consume_direct = nested_consume_columns(name, to_df_trait, ty, &series_direct);
    let consume_take = nested_consume_columns(name, to_df_trait, ty, &series_take);
    let consume_empty = nested_consume_columns(name, to_df_trait, ty, &series_empty);
    let consume_all_absent = nested_consume_columns(name, to_df_trait, ty, &series_all_absent);

    let (validity_freeze, offsets_freeze) = hoisted_freezes(shape, layers, kind, pa_root);

    if shape.has_inner_option() {
        // 4-branch dispatch. The offsets-buffer freeze is emitted inside
        // each arm rather than hoisted above the dispatch: with the freeze
        // local to each branch, LLVM specializes register allocation around
        // the heavily-inlined `columnar_from_refs` on the hot direct/take
        // paths instead of treating the `OffsetsBuffer` construction as a
        // global obligation.
        quote! {
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
        }
    } else {
        // No inner-Option: total == flat.len(). Two branches: empty when
        // flat is empty (no leaves), direct otherwise. Same
        // freeze-inside-branch rationale as the four-arm case above.
        quote! {
            #validity_freeze
            if #flat.is_empty() {
                #offsets_freeze
                #consume_empty
            } else {
                let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
                #offsets_freeze
                #consume_direct
            }
        }
    }
}

/// Build the per-element-push storage decls + leaf-body + post-scan
/// materialization scope. Wraps everything in
/// `let __df_derive_field_series_<idx>: Series = { ... };` so the caller
/// can splice it as a single Series-producing block.
#[allow(clippy::too_many_arguments)]
fn pep_emit(
    pep: &super::leaf_kind::PerElementPush,
    access: &TokenStream,
    series_local: &syn::Ident,
    shape: &VecLayers,
    layers: &[LayerIdents],
    kind: &LeafKind<'_>,
    layer_counters: &[syn::Ident],
    total: &syn::Ident,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let leaf_bind = idents::leaf_value();
    let precount = build_precount(
        access,
        shape,
        layers,
        kind.precount_outer_some_prefix(),
        total,
        layer_counters,
    );
    let leaf_body = pep_leaf_body(shape, &leaf_bind, &pep.per_elem_push);
    let scan = build_scan(
        access,
        shape,
        layers,
        kind.scan_outer_some_prefix(),
        &leaf_body,
        &pep.leaf_offsets_post_push,
    );

    let counter_for_depth = |i: usize| idents::vec_layer_total_token(i);
    let offsets_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.validity_mb).collect();
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls =
        shape_validity_decls(shape, &validity_idents, &counter_for_depth, pa_root);

    let materialize = pep_materialize(pep, shape, layers, kind, pa_root, pp);
    let storage_decls = &pep.storage_decls;
    let extra_imports = &pep.extra_imports;

    quote! {
        let #series_local: #pp::Series = {
            #extra_imports
            #precount
            #storage_decls
            #offsets_decls
            #validity_decls
            #scan
            #materialize
        };
    }
}

/// Build the collect-then-bulk emit body. The collect-then-bulk path emits
/// directly into `columns` rather than producing a single Series local —
/// the for-loop over inner schema columns runs `columns.push(...)` per
/// inner column. Returns the entire `{ ... }` block.
#[allow(clippy::too_many_arguments)]
fn ctb_emit(
    ctb: &CollectThenBulk<'_>,
    access: &TokenStream,
    shape: &VecLayers,
    layers: &[LayerIdents],
    kind: &LeafKind<'_>,
    layer_counters: &[syn::Ident],
    total: &syn::Ident,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let flat = idents::nested_flat(ctb.idx);
    let positions = idents::nested_positions(ctb.idx);
    let ty = ctb.ty;

    let precount = build_precount(
        access,
        shape,
        layers,
        kind.precount_outer_some_prefix(),
        total,
        layer_counters,
    );
    let leaf_body = ctb_leaf_body(shape, &flat, &positions);
    let leaf_offsets_post_push = if shape.has_inner_option() {
        quote! { #positions.len() }
    } else {
        quote! { #flat.len() }
    };
    let scan = build_scan(
        access,
        shape,
        layers,
        kind.scan_outer_some_prefix(),
        &leaf_body,
        &leaf_offsets_post_push,
    );

    let counter_for_depth = |i: usize| idents::nested_layer_total_token(i);
    let offsets_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.validity_mb).collect();
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls =
        shape_validity_decls(shape, &validity_idents, &counter_for_depth, pa_root);

    let positions_decl = if shape.has_inner_option() {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total);
        }
    } else {
        TokenStream::new()
    };

    let materialize = ctb_materialize(ctb, shape, layers, kind, pa_root, pp);

    quote! {{
        #precount
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::with_capacity(#total);
        #positions_decl
        #offsets_decls
        #validity_decls
        #scan
        #materialize
    }}
}

/// Unified depth-N `Vec`-bearing emitter. Produces:
///
/// - For [`LeafKind::PerElementPush`]: a `let __df_derive_field_series_<idx>: Series = { ... };`
///   declaration plus the `with_name(...)` rename and `columns.push(...)`,
///   wrapped in a `{ ... }` block scoping the per-field intermediates.
/// - For [`LeafKind::CollectThenBulk`]: a `{ ... }` block that scans the
///   field, dispatches on `(total, flat.len())`, and per inner schema
///   column pushes a list-wrapped Series onto `columns`.
///
/// Both shapes replace the original `vec::vec_emit_decl` and the inlined
/// body that was in `nested::build_nested_encoder` for the
/// `WrapperShape::Vec` arm.
pub(super) fn vec_emit_general(
    kind: &LeafKind<'_>,
    access: &TokenStream,
    idx: usize,
    shape: &VecLayers,
) -> TokenStream {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    let pp = crate::codegen::polars_paths::prelude();
    let depth = shape.depth();

    // Per-leaf-kind layer-ident factory: per-element-push uses flat-vec
    // idents (no field namespacing); collect-then-bulk uses per-(field,
    // layer) namespacing.
    let field_idx = match kind {
        LeafKind::PerElementPush(_) => None,
        LeafKind::CollectThenBulk(_) => Some(idx),
    };
    let layers: Vec<LayerIdents> = (0..depth).map(|i| layer_idents(field_idx, i)).collect();

    let total = match kind {
        LeafKind::PerElementPush(_) => idents::total_leaves(),
        LeafKind::CollectThenBulk(_) => idents::nested_total(idx),
    };
    let layer_counters: Vec<syn::Ident> = match kind {
        LeafKind::PerElementPush(_) => (0..depth.saturating_sub(1))
            .map(idents::vec_layer_total)
            .collect(),
        LeafKind::CollectThenBulk(_) => (0..depth.saturating_sub(1))
            .map(idents::nested_layer_total)
            .collect(),
    };

    match kind {
        LeafKind::PerElementPush(pep) => {
            let series_local = idents::vec_field_series(idx);
            pep_emit(
                pep,
                access,
                &series_local,
                shape,
                &layers,
                kind,
                &layer_counters,
                &total,
                &pa_root,
                &pp,
            )
        }
        LeafKind::CollectThenBulk(ctb) => ctb_emit(
            ctb,
            access,
            shape,
            &layers,
            kind,
            &layer_counters,
            &total,
            &pa_root,
            &pp,
        ),
    }
}
