//! Unified shape-aware emitter for vector-backed leaves.
//!
//! The shape-aware emitters ([`vec_emit_primitive`] and [`vec_emit_ctb`]) tie together
//! the depth-N walker primitives and diverge only at leaf storage/materialization.
//!
//! The collect-then-bulk path also accepts the depth-0 (`Leaf`) wrapper —
//! a bare nested struct or a single/multi-`Option<Nested>` — and routes it
//! through the same scan-and-materialize machinery the depth-N path uses,
//! degenerating the list-array stack to a direct Series clone (`layers
//! is_empty`).
//!
use proc_macro2::TokenStream;
use quote::quote;

use crate::codegen::encode_plan::{
    EmitOp, EncodePlan, FinishGroup, InitOp, PostScanOp, ScanOp, SeriesPlan,
};
use crate::codegen::planner::PrimitiveListPlan;
use crate::ir::{AccessChain, VecLayers, WrapperShape};

use super::idents::{self, LayerIdents};
use super::leaf_kind::{CollectThenBulk, PrimitiveListCommon};
use super::nested_columns::{
    NestedMaterializeCtx, NestedWrapper, SharedListPrefix, materialize_nested_columns,
};
use super::shape_walk::{
    ListAssembly, ListAssemblySeed, ListAssemblyTarget, ShapeEmitter, ShapeEmitterParts,
    shape_assemble_list_stack,
};
use super::{access_chain_to_ref, collapse_options_to_ref, idx_size_len_expr};
use crate::codegen::external_paths::ExternalPaths;

fn layer_idents(field_idx: usize, nested: bool, layer_idx: usize) -> LayerIdents {
    let namespace = if nested {
        idents::LayerNamespace::Nested { field_idx }
    } else {
        idents::LayerNamespace::Vec { field_idx }
    };
    LayerIdents::new(namespace, layer_idx)
}

fn element_leaf_body<'a>(
    shape: &'a VecLayers,
    leaf_bind: &'a syn::Ident,
    write_leaf: &'a TokenStream,
    prepare_segment: Option<&'a TokenStream>,
) -> impl Fn(&TokenStream) -> TokenStream + 'a {
    move |vec_bind: &TokenStream| -> TokenStream {
        let segment_prelude = prepare_segment.map(|prepare_segment| {
            let additional = idents::leaf_reserve_len();
            quote! {
                let #additional: usize = #vec_bind.len();
                #prepare_segment
            }
        });
        if shape.inner_access.is_empty() || shape.inner_access.is_single_plain_option() {
            quote! {
                #segment_prelude
                for #leaf_bind in #vec_bind.iter() {
                    #write_leaf
                }
            }
        } else {
            let raw_bind = idents::leaf_value_raw();
            let chain_ref = access_chain_to_ref(&quote! { #raw_bind }, &shape.inner_access);
            let resolved = chain_ref.expr;
            if chain_ref.has_option {
                quote! {
                    #segment_prelude
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind: ::std::option::Option<_> = #resolved;
                        #write_leaf
                    }
                }
            } else {
                quote! {
                    #segment_prelude
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind = #resolved;
                        #write_leaf
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
        let additional = idents::leaf_reserve_len();
        let positions_reserve = shape.has_inner_option().then(|| {
            quote! { #positions.reserve(#additional); }
        });
        let reserve = quote! {
            let #additional: usize = #vec_bind.len();
            #flat.reserve(#additional);
            #positions_reserve
        };
        if shape.inner_access.is_empty() {
            quote! {
                #reserve
                for #v in #vec_bind.iter() {
                    #flat.push(#v);
                }
            }
        } else if shape.inner_access.is_single_plain_option() {
            let flat_idx = idx_size_len_expr(flat, pp);
            quote! {
                #reserve
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
                    #reserve
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
                    #reserve
                    for #raw_bind in #vec_bind.iter() {
                        let #v = #resolved;
                        #flat.push(#v);
                    }
                }
            }
        }
    }
}

fn primitive_list_materialize(
    common: &PrimitiveListCommon,
    emitter: &ShapeEmitter<'_>,
    idx: usize,
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
    let arr_id_for_layer = |layer| idents::vec_layer_list_arr(idx, layer);
    let stack = shape_assemble_list_stack(ListAssembly {
        seed: ListAssemblySeed {
            payload: seed,
            arrow_dtype: seed_dtype,
            logical_dtype: common.leaf_logical_dtype.clone(),
        },
        layers: &wrap_layers,
        target: ListAssemblyTarget::from(&common.materialization),
        pp,
        pa_root,
        arr_id_for_layer: &arr_id_for_layer,
    });
    let leaf_arr_expr = &common.leaf_arr_expr;
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
    prefix: Option<SharedListPrefix<'_>>,
    paths: &ExternalPaths,
) -> TokenStream {
    let CollectThenBulk {
        row_capacity: _,
        ident_scope,
        sink,
        ty,
        columnar_trait,
        columnar_spec_trait,
        idx,
    } = *ctb;
    let flat = idents::nested_flat(idx);
    let positions = idents::nested_positions(idx);
    let (nested_wrapper, positions, total_len) = match wrapper {
        WrapperShape::Leaf(shape) if shape.is_bare() => {
            (NestedWrapper::None, None, quote! { #flat.len() })
        }
        WrapperShape::Leaf(_) => (
            NestedWrapper::None,
            Some(&positions),
            quote! { #positions.len() },
        ),
        WrapperShape::Vec(shape) => (
            NestedWrapper::List {
                shape,
                layers,
                arr_id_for_layer: idents::nested_layer_list_arr,
            },
            shape.has_inner_option().then_some(&positions),
            if shape.has_inner_option() {
                quote! { #positions.len() }
            } else {
                quote! { #flat.len() }
            },
        ),
    };

    materialize_nested_columns(&NestedMaterializeCtx {
        field_idx: idx,
        ident_scope,
        sink,
        ty,
        flat: &flat,
        positions,
        total_len,
        wrapper: nested_wrapper,
        prefix,
        columnar_trait,
        columnar_spec_trait,
        paths,
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn primitive_list_emit(
    encoding: &super::leaf_kind::PrimitiveListEncoding,
    access: &TokenStream,
    shape: &VecLayers,
    layers: &[LayerIdents],
    pa_root: &TokenStream,
    pp: &TokenStream,
    idx: usize,
) -> SeriesPlan {
    let leaf_bind = idents::leaf_value();
    let common = encoding.common();
    let emitter = ShapeEmitter::vec(ShapeEmitterParts {
        row_capacity: &common.row_capacity,
        shape,
        access,
        layers,
        pp,
        pa_root,
    });
    let offsets_decls = emitter.offsets_decls();
    let validity_decls = emitter.validity_decls();
    let materialize = primitive_list_materialize(common, &emitter, idx, pp);
    let storage_decls = &common.storage_decls;
    let extra_imports = &common.extra_imports;

    match encoding {
        PrimitiveListPlan::StreamReserved(plan) => {
            let leaf_body = element_leaf_body(
                shape,
                &leaf_bind,
                &plan.write_leaf,
                Some(&plan.prepare_segment),
            );
            let push = emitter.row_push(&leaf_body, &plan.leaf_offsets_post_push);
            SeriesPlan::new(
                vec![InitOp::new(quote! {
                    #extra_imports
                    #storage_decls
                    #offsets_decls
                    #validity_decls
                })],
                ScanOp::new(push),
                Vec::new(),
                quote! {{ #materialize }},
                common.materialization.output_slot().clone(),
            )
        }
        PrimitiveListPlan::BulkSegments(plan) => {
            let binding = &plan.binding;
            let write = &plan.write;
            let leaf_body = |vec_bind: &TokenStream| {
                quote! {
                    let #binding: &::std::vec::Vec<_> = #vec_bind;
                    #write
                }
            };
            let push = emitter.row_push(&leaf_body, &plan.leaf_offsets_post_push);
            SeriesPlan::new(
                vec![InitOp::new(quote! {
                    #extra_imports
                    #storage_decls
                    #offsets_decls
                    #validity_decls
                })],
                ScanOp::new(push),
                Vec::new(),
                quote! {{ #materialize }},
                common.materialization.output_slot().clone(),
            )
        }
        PrimitiveListPlan::ReplayRows(plan) => {
            let shape_counts = &plan.shape_counts;
            let row = &plan.row;
            let replay = &plan.replay;
            let push = emitter.row_count(shape_counts);
            let fill_segment = element_leaf_body(shape, &leaf_bind, &plan.write_leaf, None);
            let fill_row = emitter.row_push(&fill_segment, &plan.leaf_offsets_post_push);
            let exact_offsets_decls = emitter.exact_offsets_decls(shape_counts);
            let exact_validity_decls = emitter.exact_validity_decls(shape_counts);
            let cardinality_count = shape.depth() + 1;
            SeriesPlan::new(
                vec![InitOp::new(quote! {
                    #extra_imports
                    let mut #shape_counts: [usize; #cardinality_count] =
                        [0; #cardinality_count];
                })],
                ScanOp::new(push),
                vec![PostScanOp::replay_rows(quote! {
                    #storage_decls
                    #exact_offsets_decls
                    #exact_validity_decls
                    for #row in #replay {
                        #fill_row
                    }
                })],
                quote! {{ #materialize }},
                common.materialization.output_slot().clone(),
            )
        }
        PrimitiveListPlan::CaptureSegments(plan) => {
            let leaf_count = &plan.leaf_count;
            let leaf_segments = &plan.leaf_segments;
            let leaf_segment = &plan.leaf_segment;
            let collect_segment = |vec_bind: &TokenStream| {
                quote! {
                    let #leaf_segment: &::std::vec::Vec<_> = #vec_bind;
                    #leaf_count = #leaf_count.checked_add(#leaf_segment.len()).ok_or_else(||
                        #pp::polars_err!(
                            ComputeError:
                            "df-derive: flattened list element count exceeds usize range",
                        )
                    )?;
                    if !#leaf_segment.is_empty() {
                        #leaf_segments.push(#leaf_segment);
                    }
                }
            };
            let push = emitter.row_push(&collect_segment, &quote! { #leaf_count });
            let fill_segment = element_leaf_body(shape, &leaf_bind, &plan.write_leaf, None);
            let fill_segment = fill_segment(&quote! { #leaf_segment });
            let fill_leaf_storage = quote! {
                for #leaf_segment in #leaf_segments {
                    #fill_segment
                }
            };
            SeriesPlan::new(
                vec![InitOp::new(quote! {
                    #extra_imports
                    let mut #leaf_count: usize = 0;
                    let mut #leaf_segments: ::std::vec::Vec<_> =
                        ::std::vec::Vec::new();
                    #offsets_decls
                    #validity_decls
                })],
                ScanOp::new(push),
                vec![PostScanOp::new(quote! {
                    #storage_decls
                    #fill_leaf_storage
                })],
                quote! {{ #materialize }},
                common.materialization.output_slot().clone(),
            )
        }
    }
}

fn ctb_leaf_row_push_depth0(
    access: &TokenStream,
    flat: &syn::Ident,
    positions: &syn::Ident,
    option_layers: usize,
    access_chain: &AccessChain,
    pp: &TokenStream,
) -> TokenStream {
    let v = idents::leaf_value();
    if option_layers == 0 {
        let value_ref = ctb_depth0_ref_expr(access, access_chain);
        quote! { #flat.push(#value_ref); }
    } else {
        // `option_layers == 1`: match `&Option<T>` directly. `>= 2`:
        // collapse to `Option<&T>` first, then match by value. Mirrors
        // `ShapeScan::build_layer`'s opt_layers branch on outer-Vec layers.
        let match_expr = ctb_depth0_match_expr(access, access_chain, option_layers);
        let flat_idx = idx_size_len_expr(flat, pp);
        quote! {
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

#[allow(clippy::too_many_arguments)]
fn ctb_emit(
    ctb: &CollectThenBulk<'_>,
    access: &TokenStream,
    wrapper: &WrapperShape,
    layers: &[LayerIdents],
    pa_root: &TokenStream,
    pp: &TokenStream,
    prefix: Option<SharedListPrefix<'_>>,
    paths: &ExternalPaths,
) -> EncodePlan {
    let flat = idents::nested_flat(ctb.idx);
    let positions = idents::nested_positions(ctb.idx);
    let ty = ctb.ty;
    let row_capacity = ctb.row_capacity;

    let (push, offsets_decls, validity_decls) = match wrapper {
        WrapperShape::Leaf(shape) if shape.is_bare() => {
            let empty_access = AccessChain::empty();
            let push = ctb_leaf_row_push_depth0(access, &flat, &positions, 0, &empty_access, pp);
            (push, TokenStream::new(), TokenStream::new())
        }
        WrapperShape::Leaf(shape) => {
            let access_chain = shape.access();
            let push = ctb_leaf_row_push_depth0(
                access,
                &flat,
                &positions,
                access_chain.option_layers(),
                access_chain,
                pp,
            );
            (push, TokenStream::new(), TokenStream::new())
        }
        WrapperShape::Vec(shape) => {
            let emitter = ShapeEmitter::nested(ShapeEmitterParts {
                row_capacity,
                shape,
                access,
                layers,
                pp,
                pa_root,
            });
            let leaf_body = ctb_leaf_body(shape, &flat, &positions, pp);
            let leaf_offsets_post_push = if shape.has_inner_option() {
                quote! { #positions.len() }
            } else {
                quote! { #flat.len() }
            };
            let push = emitter.row_push(&leaf_body, &leaf_offsets_post_push);
            let offsets_decls = emitter.offsets_decls();
            let validity_decls = emitter.validity_decls();
            (push, offsets_decls, validity_decls)
        }
    };

    // `positions` is needed whenever any row can be absent: at depth 0 with
    // any outer Option, or at depth >= 1 with an inner Option above the leaf.
    let needs_positions = match wrapper {
        WrapperShape::Leaf(shape) => !shape.is_bare(),
        WrapperShape::Vec(shape) => shape.has_inner_option(),
    };
    let positions_decl = if needs_positions {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#row_capacity);
        }
    } else {
        TokenStream::new()
    };

    let materialize = ctb_materialize(ctb, wrapper, layers, prefix, paths);

    EncodePlan::new(
        vec![InitOp::new(quote! {
            let mut #flat: ::std::vec::Vec<&#ty> =
                ::std::vec::Vec::with_capacity(#row_capacity);
            #positions_decl
            #offsets_decls
            #validity_decls
        })],
        vec![ScanOp::new(push)],
        vec![FinishGroup::inline(
            Vec::new(),
            vec![EmitOp::new(materialize)],
        )],
    )
}

/// Shape-aware emitter for primitive `Vec` leaves. The signature requires a
/// [`VecLayers`] shape, so a primitive-list encoding cannot be paired with a
/// leaf-only wrapper.
pub(super) fn vec_emit_primitive(
    encoding: &super::leaf_kind::PrimitiveListEncoding,
    access: &TokenStream,
    idx: usize,
    shape: &VecLayers,
    paths: &ExternalPaths,
) -> SeriesPlan {
    let pa_root = paths.polars_arrow_root();
    let pp = paths.prelude();
    let depth = shape.depth();
    let layers: Vec<LayerIdents> = (0..depth).map(|i| layer_idents(idx, false, i)).collect();
    primitive_list_emit(encoding, access, shape, &layers, pa_root, pp, idx)
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
) -> EncodePlan {
    vec_emit_ctb_with_prefix(ctb, access, idx, wrapper, None, paths)
}

pub(super) fn vec_emit_ctb_with_prefix(
    ctb: &CollectThenBulk<'_>,
    access: &TokenStream,
    idx: usize,
    wrapper: &WrapperShape,
    prefix: Option<SharedListPrefix<'_>>,
    paths: &ExternalPaths,
) -> EncodePlan {
    let pa_root = paths.polars_arrow_root();
    let pp = paths.prelude();
    let depth = wrapper.vec_depth();
    let layers: Vec<LayerIdents> = (0..depth).map(|i| layer_idents(idx, true, i)).collect();
    ctb_emit(ctb, access, wrapper, &layers, pa_root, pp, prefix, paths)
}
