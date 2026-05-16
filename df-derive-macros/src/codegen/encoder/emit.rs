//! Unified shape-aware emitter for vector-backed leaves.
//!
//! The shape-aware emitters ([`vec_emit_pep`] and [`vec_emit_ctb`]) tie together
//! the depth-N walker primitives and diverge only at leaf storage/materialization.
//!
//! The collect-then-bulk path also accepts the depth-0 (`Leaf`) wrapper —
//! a bare nested struct or a single/multi-`Option<Nested>` — and routes it
//! through the same scan-and-materialize machinery the depth-N path uses,
//! degenerating the list-array stack to a direct Series clone (`layers
//! is_empty`) and using `items.len()` rather than the precount `total` for
//! the all-absent arm length (precount has no leaves to count at depth 0).
//!
use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::{AccessChain, LeafShape, VecLayers, WrapperShape};

use super::idents::{self, LayerIdents};
use super::leaf_kind::CollectThenBulk;
use super::nested_columns::{NestedMaterializeCtx, NestedWrapper, materialize_nested_columns};
use super::shape_walk::{ShapeEmitter, ShapeEmitterParts, shape_assemble_list_stack};
use super::{access_chain_to_ref, collapse_options_to_ref, idx_size_len_expr};
use crate::codegen::external_paths::ExternalPaths;

fn layer_idents(field_idx: Option<usize>, layer_idx: usize) -> LayerIdents {
    let namespace = field_idx.map_or(idents::LayerNamespace::Vec, |idx| {
        idents::LayerNamespace::Nested { field_idx: idx }
    });
    LayerIdents::new(namespace, layer_idx)
}

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

fn pep_materialize(
    pep: &super::leaf_kind::PerElementPush,
    emitter: &ShapeEmitter<'_>,
    pp: &TokenStream,
) -> TokenStream {
    let pa_root = emitter.pa_root;
    let leaf_arr = idents::leaf_arr();
    let seed_arrow_dtype_id = idents::seed_arrow_dtype();
    let seed_dtype_decl = quote! {
        let #seed_arrow_dtype_id: #pa_root::datatypes::ArrowDataType =
            #pa_root::array::Array::dtype(&#leaf_arr).clone();
    };
    let seed = quote! { ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef };
    let seed_dtype = quote! { #seed_arrow_dtype_id };
    let wrap_layers = emitter.layer_wraps_move();
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

fn ctb_materialize(
    ctb: &CollectThenBulk<'_>,
    wrapper: &WrapperShape,
    layers: &[LayerIdents],
    paths: &ExternalPaths,
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
    let total = idents::nested_total(idx);

    let (nested_wrapper, positions, total_len) = match wrapper {
        WrapperShape::Leaf(LeafShape::Bare) => (NestedWrapper::None, None, quote! { items.len() }),
        WrapperShape::Leaf(LeafShape::Optional { .. }) => (
            NestedWrapper::None,
            Some(&positions),
            quote! { items.len() },
        ),
        WrapperShape::Vec(shape) => (
            NestedWrapper::List {
                shape,
                layers,
                arr_id_for_layer: idents::nested_layer_list_arr,
            },
            shape.has_inner_option().then_some(&positions),
            quote! { #total },
        ),
    };

    materialize_nested_columns(&NestedMaterializeCtx {
        field_idx: idx,
        ty,
        column_prefix: name,
        flat: &flat,
        positions,
        total_len,
        wrapper: nested_wrapper,
        columnar_trait,
        to_df_trait,
        paths,
    })
}

#[allow(clippy::too_many_arguments)]
fn pep_emit(
    pep: &super::leaf_kind::PerElementPush,
    access: &TokenStream,
    series_local: &syn::Ident,
    shape: &VecLayers,
    layers: &[LayerIdents],
    layer_counters: &[syn::Ident],
    total: &syn::Ident,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let leaf_bind = idents::leaf_value();
    let emitter = ShapeEmitter::vec(ShapeEmitterParts {
        shape,
        access,
        layers,
        total_counter: total,
        layer_counters,
        pp,
        pa_root,
    });
    let precount = emitter.precount();
    let leaf_body = pep_leaf_body(shape, &leaf_bind, &pep.per_elem_push);
    let scan = emitter.scan(&leaf_body, &pep.leaf_offsets_post_push);

    let offsets_decls = emitter.offsets_decls();
    let validity_decls = emitter.validity_decls();

    let materialize = pep_materialize(pep, &emitter, pp);
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

#[allow(clippy::too_many_arguments)]
fn ctb_emit(
    ctb: &CollectThenBulk<'_>,
    access: &TokenStream,
    wrapper: &WrapperShape,
    layers: &[LayerIdents],
    layer_counters: &[syn::Ident],
    total: &syn::Ident,
    pa_root: &TokenStream,
    pp: &TokenStream,
    paths: &ExternalPaths,
) -> TokenStream {
    let flat = idents::nested_flat(ctb.idx);
    let positions = idents::nested_positions(ctb.idx);
    let ty = ctb.ty;

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
            let emitter = ShapeEmitter::nested(ShapeEmitterParts {
                shape,
                access,
                layers,
                total_counter: total,
                layer_counters,
                pp,
                pa_root,
            });
            let precount = emitter.precount();
            let leaf_body = ctb_leaf_body(shape, &flat, &positions, pp);
            let leaf_offsets_post_push = if shape.has_inner_option() {
                quote! { #positions.len() }
            } else {
                quote! { #flat.len() }
            };
            let scan = emitter.scan(&leaf_body, &leaf_offsets_post_push);
            let offsets_decls = emitter.offsets_decls();
            let validity_decls = emitter.validity_decls();
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

    let materialize = ctb_materialize(ctb, wrapper, layers, paths);

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
        &layer_counters,
        &total,
        pa_root,
        pp,
        paths,
    )
}
