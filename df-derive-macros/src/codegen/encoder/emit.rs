//! Unified shape-aware emitter parameterized by [`LeafKind`].
//!
//! [`vec_emit_general`] is the single place that ties together the
//! depth-N walker primitives ([`ShapePrecount`], [`ShapeScan`],
//! [`shape_offsets_decls`], [`shape_validity_decls`],
//! [`shape_assemble_list_stack`]). It dispatches on [`LeafKind`] for the
//! points where per-element-push and collect-then-bulk genuinely diverge.
//!
//! The collect-then-bulk path also accepts the depth-0 (`Leaf`) wrapper —
//! a bare nested struct or a single/multi-`Option<Nested>` — and routes it
//! through the same scan-and-materialize machinery the depth-N path uses,
//! degenerating the list-array stack to a direct Series clone (`layers
//! is_empty`) and using `items.len()` rather than the precount `total` for
//! the all-absent arm length (precount has no leaves to count at depth 0).
//!
//! See `docs/encoder-ir.md` for the conceptual model.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::{AccessChain, LeafShape, VecLayers, WrapperShape};

use super::idents;
use super::leaf_kind::{CollectThenBulk, LeafKind};
use super::shape_walk::{
    LayerIdents, LayerWrap, ShapePrecount, ShapeScan, freeze_offsets_buf, freeze_validity_bitmap,
    shape_assemble_list_stack, shape_offsets_decls, shape_validity_decls,
};
use super::{access_chain_to_ref, collapse_options_to_ref, idx_size_len_expr};
use crate::codegen::external_paths::ExternalPaths;

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
/// fully qualified offsets conversion plus optional `Bitmap::from(...)`
/// for the per-element-push path (freezes interleaved with each wrap).
fn layer_wraps<'a>(
    shape: &VecLayers,
    layers: &'a [LayerIdents],
    kind: &LeafKind,
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
            let mut fd = freeze_offsets_buf(buf_id, &layer.offsets, pa_root);
            if validity_bm.is_some() {
                fd.extend(freeze_validity_bitmap(
                    &layer.validity_bm,
                    &layer.validity_mb,
                    pa_root,
                ));
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
    kind: &LeafKind,
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
        validity_freeze.push(freeze_validity_bitmap(
            &layer.validity_bm,
            &layer.validity_mb,
            pa_root,
        ));
    }
    let mut offsets_freeze: Vec<TokenStream> = Vec::new();
    for layer in layers {
        offsets_freeze.push(freeze_offsets_buf(
            &layer.offsets_buf,
            &layer.offsets,
            pa_root,
        ));
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
        projection: None,
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
    pp: &TokenStream,
) -> TokenStream {
    ShapeScan {
        shape,
        access,
        layers,
        outer_some_prefix,
        leaf_body,
        leaf_offsets_post_push,
        pp,
        projection: None,
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
        if shape.inner_access.is_empty() || shape.inner_access.is_single_plain_option() {
            quote! {
                for #leaf_bind in #vec_bind.iter() {
                    #per_elem_push
                }
            }
        } else {
            let raw_bind = idents::leaf_value_raw();
            let chain_ref = access_chain_to_ref(&quote! { #raw_bind }, &shape.inner_access);
            let resolved = chain_ref.expr;
            if chain_ref.has_option {
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind: ::std::option::Option<_> = #resolved;
                        #per_elem_push
                    }
                }
            } else {
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind = #resolved;
                        #per_elem_push
                    }
                }
            }
        }
    }
}

fn ctb_depth0_match_expr(
    access: &TokenStream,
    access_chain: &AccessChain,
    option_layers: usize,
) -> TokenStream {
    if access_chain.is_single_plain_option() {
        quote! { &(#access) }
    } else if access_chain.is_only_options() {
        collapse_options_to_ref(access, option_layers)
    } else {
        access_chain_to_ref(&quote! { &(#access) }, access_chain).expr
    }
}

fn ctb_depth0_ref_expr(access: &TokenStream, access_chain: &AccessChain) -> TokenStream {
    if access_chain.is_empty() {
        quote! { &(#access) }
    } else {
        access_chain_to_ref(&quote! { &(#access) }, access_chain).expr
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
    pp: &'a TokenStream,
) -> impl Fn(&TokenStream) -> TokenStream + 'a {
    move |vec_bind: &TokenStream| -> TokenStream {
        let maybe = idents::nested_maybe();
        let v = idents::leaf_value();
        if shape.inner_access.is_empty() {
            quote! {
                for #v in #vec_bind.iter() {
                    #flat.push(#v);
                }
            }
        } else if shape.inner_access.is_single_plain_option() {
            let flat_idx = idx_size_len_expr(flat, pp);
            quote! {
                for #maybe in #vec_bind.iter() {
                    match #maybe {
                        ::std::option::Option::Some(#v) => {
                            #positions.push(::std::option::Option::Some(
                                #flat_idx,
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
            let raw_bind = idents::leaf_value_raw();
            let chain_ref = access_chain_to_ref(&quote! { #raw_bind }, &shape.inner_access);
            let resolved = chain_ref.expr;
            if chain_ref.has_option {
                let flat_idx = idx_size_len_expr(flat, pp);
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        match #resolved {
                            ::std::option::Option::Some(#v) => {
                                #positions.push(::std::option::Option::Some(
                                    #flat_idx,
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
                    for #raw_bind in #vec_bind.iter() {
                        let #v = #resolved;
                        #flat.push(#v);
                    }
                }
            }
        }
    }
}

/// Build the per-column emit body that iterates `<T as ToDataFrame>::schema()?`
/// and pushes each inner-Series-yielding expression onto `columns` with the
/// parent name prefixed. Shared by every dispatch arm of [`ctb_materialize`]
/// (depth-0 direct/take/null and depth-N empty/direct/take/all-absent), with
/// each arm supplying a different per-column inner-Series expression.
fn nested_consume_columns(
    parent_name: &str,
    to_df_trait: &TokenStream,
    ty: &TokenStream,
    series_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
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
///
/// At depth 0 (`WrapperShape::Leaf`) the helper degenerates to the
/// `inner_col_expr` unchanged — the depth-0 shape (bare/option `Nested`) has
/// no list layers, so rechunking and stacking would just round-trip the
/// inner Series through `chunks()[0]`.
fn ctb_layer_wrap(
    wrapper: &WrapperShape,
    layers: &[LayerIdents],
    kind: &LeafKind,
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
    pa_root: &TokenStream,
) -> TokenStream {
    let WrapperShape::Vec(shape) = wrapper else {
        return inner_col_expr.clone();
    };
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
        pp,
        pa_root,
        &idents::nested_layer_list_arr,
    );
    quote! {{
        #chunk_decl
        #stack
    }}
}

/// Materialize the post-scan tokens for the per-element-push leaf kind:
/// build the typed leaf array, then chain the depth-N `LargeListArray::new`
/// wraps with offsets/validity freezes interleaved per layer.
fn pep_materialize(
    pep: &super::leaf_kind::PerElementPush,
    shape: &VecLayers,
    layers: &[LayerIdents],
    kind: &LeafKind,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    // Capture the leaf dtype before boxing because boxing moves the typed
    // array into the list stack.
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
        pp,
        pa_root,
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
/// would waste work), branch on `(total, flat.len())` to 1, 2, 3, or 4
/// arms (depending on shape), and per arm iterate the inner schema to wrap
/// each per-column inner Series in N `LargeListArray::new` layers (zero
/// layers at depth 0 — see [`ctb_layer_wrap`]).
///
/// Branch shapes:
/// - depth 0, bare (`Leaf { option_layers: 0 }`): 1 branch — direct
///   (every row contributes one leaf, no per-row Option to skip).
/// - depth 0, option (`Leaf { option_layers >= 1 }`): 3 branches —
///   `flat.is_empty()` (every row was None; typed-null Series of length
///   `items.len()`), `flat.len() == items.len()` (every row was Some;
///   direct), else mixed (`IdxCa::take` per column).
/// - depth >= 1, no inner-Option: 2 branches — empty (no leaves) or direct.
/// - depth >= 1, with inner-Option: 4 branches — `total == 0` (no leaf
///   slots), `flat.is_empty() && total > 0` (all leaves None; typed-null
///   Series of length `total`), `flat.len() == total` (all Some; direct),
///   else mixed (`IdxCa::take` per column).
#[allow(clippy::too_many_lines)]
fn ctb_materialize(
    ctb: &CollectThenBulk<'_>,
    wrapper: &WrapperShape,
    layers: &[LayerIdents],
    kind: &LeafKind,
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
    let validate_nested_frame = idents::validate_nested_frame();
    let validate_nested_column_dtype = idents::validate_nested_column_dtype();

    // Decls reused across match arms.
    let df_decl = quote! {
        let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
        #validate_nested_frame(&#df, #flat.len(), ::core::any::type_name::<#ty>())?;
    };
    let take_decl = quote! {
        let #take: #pp::IdxCa =
            <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                "".into(),
                #positions.iter().copied(),
            );
    };

    // Per-column inner-Series expressions for the four branches (or two,
    // when no inner Option). The list-array wrap is identical across
    // branches; only the seed expression differs.
    let inner_col_direct = quote! {{
        let #inner_full = #df.column(#col_name)?.as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
        #inner_full.clone()
    }};
    let inner_col_take = quote! {{
        let #inner_full = #df
            .column(#col_name)?
            .as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
        #inner_full.take(&#take)?
    }};
    let inner_col_empty = quote! {
        #pp::Series::new_empty("".into(), #dtype)
    };
    // Depth 0 has no offsets to constrain the null length, so the all-absent
    // arm there uses `items.len()` (one row per outer position). Depth >= 1
    // sizes the typed-null chunk to `total` (sum of inner-Vec lengths) so
    // the offsets buffer's max value doesn't exceed the chunk's length.
    let absent_len: TokenStream = match wrapper {
        WrapperShape::Leaf(_) => quote! { items.len() },
        WrapperShape::Vec(_) => quote! { #total },
    };
    let inner_col_all_absent = quote! {
        #pp::Series::new_empty("".into(), #dtype)
            .extend_constant(#pp::AnyValue::Null, #absent_len)?
    };

    let series_direct = ctb_layer_wrap(wrapper, layers, kind, &inner_col_direct, pp, pa_root);
    let series_take = ctb_layer_wrap(wrapper, layers, kind, &inner_col_take, pp, pa_root);
    let series_empty = ctb_layer_wrap(wrapper, layers, kind, &inner_col_empty, pp, pa_root);
    let series_all_absent =
        ctb_layer_wrap(wrapper, layers, kind, &inner_col_all_absent, pp, pa_root);

    let consume_direct = nested_consume_columns(name, to_df_trait, ty, &series_direct, pp);
    let consume_take = nested_consume_columns(name, to_df_trait, ty, &series_take, pp);
    let consume_empty = nested_consume_columns(name, to_df_trait, ty, &series_empty, pp);
    let consume_all_absent = nested_consume_columns(name, to_df_trait, ty, &series_all_absent, pp);

    let (validity_freeze, offsets_freeze) = match wrapper {
        WrapperShape::Leaf(_) => (TokenStream::new(), TokenStream::new()),
        WrapperShape::Vec(shape) => hoisted_freezes(shape, layers, kind, pa_root),
    };

    match wrapper {
        WrapperShape::Leaf(LeafShape::Bare) => {
            // Bare nested struct: every row contributes one ref. One arm.
            quote! {
                #df_decl
                #consume_direct
            }
        }
        WrapperShape::Leaf(LeafShape::Optional { .. }) => {
            // Option<...<Option<Nested>>>: 3-arm dispatch. No `total == 0`
            // arm because depth-0 has no offsets buffer to size — the empty
            // (zero-rows) and all-absent arms collapse into one
            // (`flat.is_empty()` covers both, with null length `items.len()`).
            quote! {
                if #flat.is_empty() {
                    #consume_all_absent
                } else if #flat.len() == items.len() {
                    #df_decl
                    #consume_direct
                } else {
                    #df_decl
                    #take_decl
                    #consume_take
                }
            }
        }
        WrapperShape::Vec(shape) if shape.has_inner_option() => {
            // 4-branch dispatch. Offsets are frozen only in the selected arm.
            quote! {
                #validity_freeze
                if #total == 0 {
                    #offsets_freeze
                    #consume_empty
                } else if #flat.is_empty() {
                    #offsets_freeze
                    #consume_all_absent
                } else if #flat.len() == #total {
                    #df_decl
                    #offsets_freeze
                    #consume_direct
                } else {
                    #df_decl
                    #take_decl
                    #offsets_freeze
                    #consume_take
                }
            }
        }
        WrapperShape::Vec(_) => {
            // No inner-Option: total == flat.len(). Two branches: empty when
            // flat is empty (no leaves), direct otherwise.
            quote! {
                #validity_freeze
                if #flat.is_empty() {
                    #offsets_freeze
                    #consume_empty
                } else {
                    #df_decl
                    #offsets_freeze
                    #consume_direct
                }
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
    kind: &LeafKind,
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
        pp,
    );

    let counter_for_depth = |i: usize| idents::vec_layer_total_token(i);
    let offsets_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.validity_mb).collect();
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls = shape_validity_decls(shape, &validity_idents, &counter_for_depth, pa_root);

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

/// Build the depth-0 (`WrapperShape::Leaf`) row scan for the collect-then-bulk
/// path. At depth 0 every row contributes at most one leaf, so the body is a
/// straight per-row push (`option_layers == 0`) or a per-row Option match
/// (`option_layers >= 1`) without any inner-Vec iteration. For
/// `option_layers >= 2` the access expression is pre-collapsed to
/// `Option<&T>` via [`collapse_options_to_ref`]; the depth >= 1 walker does
/// the equivalent collapse via [`ShapeScan::build_layer`].
fn ctb_leaf_scan_depth0(
    access: &TokenStream,
    flat: &syn::Ident,
    positions: &syn::Ident,
    option_layers: usize,
    access_chain: &AccessChain,
    pp: &TokenStream,
) -> TokenStream {
    let it = idents::populator_iter();
    let v = idents::leaf_value();
    if option_layers == 0 {
        let value_ref = ctb_depth0_ref_expr(access, access_chain);
        quote! {
            for #it in items {
                #flat.push(#value_ref);
            }
        }
    } else {
        // `option_layers == 1`: match `&Option<T>` directly. `>= 2`:
        // collapse to `Option<&T>` first, then match by value. Mirrors
        // `ShapeScan::build_layer`'s opt_layers branch on outer-Vec layers.
        let match_expr = ctb_depth0_match_expr(access, access_chain, option_layers);
        let flat_idx = idx_size_len_expr(flat, pp);
        quote! {
            for #it in items {
                match #match_expr {
                    ::std::option::Option::Some(#v) => {
                        #positions.push(::std::option::Option::Some(
                            #flat_idx,
                        ));
                        #flat.push(#v);
                    }
                    ::std::option::Option::None => {
                        #positions.push(::std::option::Option::None);
                    }
                }
            }
        }
    }
}

/// Build the collect-then-bulk emit body. The collect-then-bulk path emits
/// directly into `columns` rather than producing a single Series local —
/// the for-loop over inner schema columns runs `columns.push(...)` per
/// inner column. Returns the entire `{ ... }` block.
///
/// At depth 0 (`WrapperShape::Leaf`) the precount / offsets / validity decls
/// drop out and the scan runs per-row (one push per row); the dispatch in
/// [`ctb_materialize`] uses `items.len()` rather than `total` for the
/// all-absent arm length.
#[allow(clippy::too_many_arguments)]
fn ctb_emit(
    ctb: &CollectThenBulk<'_>,
    access: &TokenStream,
    wrapper: &WrapperShape,
    layers: &[LayerIdents],
    kind: &LeafKind,
    layer_counters: &[syn::Ident],
    total: &syn::Ident,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let flat = idents::nested_flat(ctb.idx);
    let positions = idents::nested_positions(ctb.idx);
    let ty = ctb.ty;

    // Per-shape scan + sizing. At depth 0 the precount loop and per-layer
    // offsets/validity decls are vacuous, so the scan is the entire driver;
    // depth >= 1 routes through the shared shape walkers.
    let (precount, scan, offsets_decls, validity_decls, flat_capacity) = match wrapper {
        WrapperShape::Leaf(LeafShape::Bare) => {
            let empty_access = AccessChain::empty();
            let scan = ctb_leaf_scan_depth0(access, &flat, &positions, 0, &empty_access, pp);
            (
                TokenStream::new(),
                scan,
                TokenStream::new(),
                TokenStream::new(),
                quote! { items.len() },
            )
        }
        WrapperShape::Leaf(LeafShape::Optional {
            option_layers,
            access: access_chain,
        }) => {
            let scan = ctb_leaf_scan_depth0(
                access,
                &flat,
                &positions,
                option_layers.get(),
                access_chain,
                pp,
            );
            (
                TokenStream::new(),
                scan,
                TokenStream::new(),
                TokenStream::new(),
                quote! { items.len() },
            )
        }
        WrapperShape::Vec(shape) => {
            let precount = build_precount(
                access,
                shape,
                layers,
                kind.precount_outer_some_prefix(),
                total,
                layer_counters,
            );
            let leaf_body = ctb_leaf_body(shape, &flat, &positions, pp);
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
                pp,
            );
            let counter_for_depth = |i: usize| idents::nested_layer_total_token(i);
            let offsets_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.offsets).collect();
            let validity_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.validity_mb).collect();
            let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
            let validity_decls =
                shape_validity_decls(shape, &validity_idents, &counter_for_depth, pa_root);
            (
                precount,
                scan,
                offsets_decls,
                validity_decls,
                quote! { #total },
            )
        }
    };

    // `positions` is needed whenever any row can be absent: at depth 0 with
    // any outer Option, or at depth >= 1 with an inner Option above the leaf.
    let needs_positions = match wrapper {
        WrapperShape::Leaf(LeafShape::Bare) => false,
        WrapperShape::Leaf(LeafShape::Optional { .. }) => true,
        WrapperShape::Vec(shape) => shape.has_inner_option(),
    };
    let positions_decl = if needs_positions {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#flat_capacity);
        }
    } else {
        TokenStream::new()
    };

    let materialize = ctb_materialize(ctb, wrapper, layers, kind, pa_root, pp);

    quote! {{
        #precount
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::with_capacity(#flat_capacity);
        #positions_decl
        #offsets_decls
        #validity_decls
        #scan
        #materialize
    }}
}

/// Shape-aware emitter for primitive `Vec` leaves. The signature requires a
/// [`VecLayers`] shape, so a per-element-push leaf cannot be paired with a
/// leaf-only wrapper.
pub(super) fn vec_emit_pep(
    pep: &super::leaf_kind::PerElementPush,
    access: &TokenStream,
    idx: usize,
    shape: &VecLayers,
    paths: &ExternalPaths,
) -> TokenStream {
    let pa_root = paths.polars_arrow_root();
    let pp = paths.prelude();
    let kind = LeafKind::PerElementPush;
    let depth = shape.depth();
    let layers: Vec<LayerIdents> = (0..depth).map(|i| layer_idents(None, i)).collect();
    let total = idents::total_leaves();
    let layer_counters: Vec<syn::Ident> = (0..depth.saturating_sub(1))
        .map(idents::vec_layer_total)
        .collect();
    let series_local = idents::vec_field_series(idx);
    pep_emit(
        pep,
        access,
        &series_local,
        shape,
        &layers,
        &kind,
        &layer_counters,
        &total,
        pa_root,
        pp,
    )
}

/// Shape-aware emitter for nested struct / generic leaves. Accepts the full
/// wrapper because collect-then-bulk supports both depth-0 leaf shapes and
/// every Vec-bearing shape.
pub(super) fn vec_emit_ctb(
    ctb: &CollectThenBulk<'_>,
    access: &TokenStream,
    idx: usize,
    wrapper: &WrapperShape,
    paths: &ExternalPaths,
) -> TokenStream {
    let pa_root = paths.polars_arrow_root();
    let pp = paths.prelude();
    let kind = LeafKind::CollectThenBulk;
    let depth = wrapper.vec_depth();
    let layers: Vec<LayerIdents> = (0..depth).map(|i| layer_idents(Some(idx), i)).collect();
    let total = idents::nested_total(idx);
    let layer_counters: Vec<syn::Ident> = (0..depth.saturating_sub(1))
        .map(idents::nested_layer_total)
        .collect();
    ctb_emit(
        ctb,
        access,
        wrapper,
        &layers,
        &kind,
        &layer_counters,
        &total,
        pa_root,
        pp,
    )
}
