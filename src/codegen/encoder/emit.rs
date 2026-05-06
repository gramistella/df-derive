//! Unified depth-N `Vec`-bearing emitter parameterized by [`LeafKind`].
//!
//! Both leaf kinds (per-element-push primitives and collect-then-bulk
//! nested structs/generics) share the depth-N walker scaffolding:
//! [`ShapePrecount`] sizes per-layer counters; [`ShapeScan`] walks the
//! per-row push body; [`shape_offsets_decls`] / [`shape_validity_decls`]
//! allocate the offsets vecs and validity bitmaps; [`shape_assemble_list_stack`]
//! chains `LargeListArray::new` per layer and routes the outermost through
//! `__df_derive_assemble_list_series_unchecked`.
//!
//! [`vec_emit_general`] is the single place that strings these primitives
//! together. It dispatches on [`LeafKind`] for the points the two paths
//! genuinely diverge:
//!
//! - storage decls before the scan
//! - per-row push body inside the scan's deepest layer
//! - post-scan materialization (one Series for per-element-push; for-loop
//!   over inner schema with 2- or 4-arm dispatch for collect-then-bulk)
//! - offsets-buffer ownership policy (Move vs Clone)
//! - freeze placement (interleaved with each wrap vs hoisted above the
//!   branch dispatch)
//!
//! The bool-d1-bare carve-out (`Vec<bool>` at depth 1, no inner Option, no
//! outer Option) lives in [`super::vec`] and bypasses this emitter — its
//! tighter `BooleanArray::from_slice` shape is a perf-driven outlier, not
//! a leaf-kind variation.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::VecLayers;

use super::collapse_options_to_ref;
use super::idents;
use super::leaf_kind::{CollectThenBulk, EmitShape, LeafKind};
use super::shape_walk::{
    LayerIdents, ShapePrecount, ShapeScan, shape_assemble_list_stack, shape_offsets_decls,
    shape_validity_decls,
};

/// Produce per-layer ident bundle for the collect-then-bulk path.
/// Per-(field, layer) namespaced so the same `idx` doesn't collide across
/// fields. The per-element-push path's matching factory lives in
/// [`super::vec`] (per-layer only — the field idx isn't needed since the
/// emit block is scoped per field via `__df_derive_field_series_<idx>`).
pub(super) fn nested_layer_idents(idx: usize, depth: usize) -> Vec<LayerIdents> {
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
    shape: &EmitShape<'_>,
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
    let wrap_layers = shape.layer_wraps(kind, pa_root);
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
    shape: &EmitShape<'_>,
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
    let wrap_layers = shape.layer_wraps(kind, pa_root);
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
    shape: &EmitShape<'_>,
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

    let series_direct = ctb_layer_wrap(shape, kind, &inner_col_direct, pp, pa_root);
    let series_take = ctb_layer_wrap(shape, kind, &inner_col_take, pp, pa_root);
    let series_empty = ctb_layer_wrap(shape, kind, &inner_col_empty, pp, pa_root);
    let series_all_absent = ctb_layer_wrap(shape, kind, &inner_col_all_absent, pp, pa_root);

    let consume_direct = nested_consume_columns(name, to_df_trait, ty, &series_direct);
    let consume_take = nested_consume_columns(name, to_df_trait, ty, &series_take);
    let consume_empty = nested_consume_columns(name, to_df_trait, ty, &series_empty);
    let consume_all_absent = nested_consume_columns(name, to_df_trait, ty, &series_all_absent);

    let (validity_freeze, offsets_freeze) = shape.hoisted_freezes(kind, pa_root);

    if shape.shape.has_inner_option() {
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
    shape: &EmitShape<'_>,
    kind: &LeafKind<'_>,
    layer_counters: &[syn::Ident],
    total: &syn::Ident,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let leaf_bind = idents::leaf_value();
    let precount = build_precount(
        access,
        shape.shape,
        shape.layers,
        kind.precount_outer_some_prefix(),
        total,
        layer_counters,
    );
    let leaf_body = pep_leaf_body(shape.shape, &leaf_bind, &pep.per_elem_push);
    let scan = build_scan(
        access,
        shape.shape,
        shape.layers,
        kind.scan_outer_some_prefix(),
        &leaf_body,
        &pep.leaf_offsets_post_push,
    );

    let counter_for_depth = |i: usize| idents::vec_layer_total_token(i);
    let offsets_idents: Vec<&syn::Ident> = shape.layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = shape.layers.iter().map(|l| &l.validity_mb).collect();
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls =
        shape_validity_decls(shape.shape, &validity_idents, &counter_for_depth, pa_root);

    let materialize = pep_materialize(pep, shape, kind, pa_root, pp);
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
    shape: &EmitShape<'_>,
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
        shape.shape,
        shape.layers,
        kind.precount_outer_some_prefix(),
        total,
        layer_counters,
    );
    let leaf_body = ctb_leaf_body(shape.shape, &flat, &positions);
    let leaf_offsets_post_push = if shape.shape.has_inner_option() {
        quote! { #positions.len() }
    } else {
        quote! { #flat.len() }
    };
    let scan = build_scan(
        access,
        shape.shape,
        shape.layers,
        kind.scan_outer_some_prefix(),
        &leaf_body,
        &leaf_offsets_post_push,
    );

    let counter_for_depth = |i: usize| idents::nested_layer_total_token(i);
    let offsets_idents: Vec<&syn::Ident> = shape.layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = shape.layers.iter().map(|l| &l.validity_mb).collect();
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls =
        shape_validity_decls(shape.shape, &validity_idents, &counter_for_depth, pa_root);

    let positions_decl = if shape.shape.has_inner_option() {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total);
        }
    } else {
        TokenStream::new()
    };

    let materialize = ctb_materialize(ctb, shape, kind, pa_root, pp);

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
/// body of `nested::nested_vec_encoder_general` (which now survives as a
/// thin shim that builds a `LeafKind::CollectThenBulk` and delegates here).
pub(super) fn vec_emit_general(
    kind: &LeafKind<'_>,
    access: &TokenStream,
    idx: usize,
    shape: &VecLayers,
) -> TokenStream {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    let pp = crate::codegen::polars_paths::prelude();
    let depth = shape.depth();

    // Per-leaf-kind layer-ident factories. The per-element-push factory
    // lives in [`super::vec`] (per-layer only); the collect-then-bulk one
    // is here (per-(field, layer)).
    let layers: Vec<LayerIdents> = match kind {
        LeafKind::PerElementPush(_) => super::vec::vec_layer_idents(depth),
        LeafKind::CollectThenBulk(_) => nested_layer_idents(idx, depth),
    };
    let emit_shape = EmitShape::new(shape, &layers);

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
                &emit_shape,
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
            &emit_shape,
            kind,
            &layer_counters,
            &total,
            &pa_root,
            &pp,
        ),
    }
}
