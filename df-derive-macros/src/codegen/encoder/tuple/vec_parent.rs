use crate::codegen::MacroConfig;
use crate::ir::{AccessChain, LeafShape, PrimitiveLeaf, TupleElement, VecLayers, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;

use super::super::shape_walk::{LayerIdents, ShapeEmitter, shape_assemble_list_stack};
use super::super::{BaseCtx, LeafCtx, access_chain_to_ref, idents};
use super::nested::emit_vec_parent_nested;
use super::projection::{
    TupleProjection, concat_access_chains, deepest_leaf_projection_access,
    prepend_parent_option_access, projected_leaf_body,
};
use super::{TupleLeafRoute, tuple_nested_type_path};

/// Emit a tuple element column with a Vec-bearing parent. Composes parent +
/// element wrappers, with the projection injected at the parent/element
/// boundary. Uses the shared shape walker with tuple-specific projection
/// layer.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_vec_parent(
    parent_access: &TokenStream,
    parent_layers: &VecLayers,
    elem: &TupleElement,
    leaf_route: TupleLeafRoute<'_>,
    elem_idx: usize,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    // Compose layers: parent + element. The element's outermost
    // `option_layers_above` (when it has its own Vec) attaches to the
    // boundary — semantically, parent's `inner_option_layers` (Options
    // immediately above the leaf, which is the projected tuple) become
    // outer-Option above the element's first Vec layer. Polars folds
    // consecutive Options into one bit, so the carry is additive.
    let mut composed_layers = parent_layers.layers.clone();
    let carried_inner_option = parent_layers.inner_option_layers;

    let composed_inner_option = match &elem.wrapper_shape {
        WrapperShape::Vec(elem_layers) => {
            let mut e_layers = elem_layers.layers.clone();
            // Parent's inner-Option (above-tuple Options) attaches to
            // element's outermost Vec as outer validity.
            e_layers[0].option_layers_above += carried_inner_option;
            e_layers[0].access =
                prepend_parent_option_access(&parent_layers.inner_access, &e_layers[0].access);
            composed_layers.extend(e_layers);
            elem_layers.inner_option_layers
        }
        WrapperShape::Leaf(leaf_shape) => {
            // No element Vec layers. Parent's inner-Option and element's
            // option_layers both attach to the leaf — Polars folds them.
            carried_inner_option + leaf_shape.option_layers()
        }
    };
    let composed_inner_access = match &elem.wrapper_shape {
        WrapperShape::Vec(elem_layers) => elem_layers.inner_access.clone(),
        WrapperShape::Leaf(LeafShape::Bare) => parent_layers.inner_access.clone(),
        WrapperShape::Leaf(LeafShape::Optional { access, .. }) => {
            concat_access_chains(&parent_layers.inner_access, access)
        }
    };

    let composed_shape = VecLayers {
        layers: composed_layers,
        inner_option_layers: composed_inner_option,
        inner_access: composed_inner_access,
    };
    let projection_layer = parent_layers.depth();
    let elem_li = syn::Index::from(elem_idx);
    let projection_path = quote! { .#elem_li };
    let projection = TupleProjection {
        layer: projection_layer,
        path: &projection_path,
        parent_access: &parent_layers.inner_access,
        smart_ptr_depth: elem.outer_smart_ptr_depth,
    };

    match leaf_route {
        TupleLeafRoute::Nested(nested) => {
            let type_path = tuple_nested_type_path(nested);
            let leaf_projection_access =
                deepest_leaf_projection_access(&composed_shape, projection, elem);
            emit_vec_parent_nested(
                parent_access,
                &composed_shape,
                projection,
                leaf_projection_access.as_ref(),
                &type_path,
                field_idx,
                column_prefix,
                config,
            )
        }
        TupleLeafRoute::Primitive(leaf) => emit_vec_parent_primitive(
            parent_access,
            &composed_shape,
            projection,
            leaf,
            elem,
            field_idx,
            column_prefix,
            config,
        ),
    }
}

/// Emit a tuple element column for a primitive element under a Vec-bearing
/// parent. Builds the per-element-push pipeline: precount, leaf storage,
/// per-layer offsets/validity, scan with projection at the boundary, and
/// post-scan stack of `LargeListArray::new` calls.
#[allow(clippy::too_many_arguments)]
fn emit_vec_parent_primitive(
    parent_access: &TokenStream,
    composed_shape: &VecLayers,
    projection: TupleProjection<'_>,
    leaf: PrimitiveLeaf<'_>,
    elem: &TupleElement,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let series_local = idents::vec_field_series(field_idx);
    let named = idents::field_named_series();
    let leaf_arr = idents::leaf_arr();
    let total_leaves = idents::total_leaves();

    let layers: Vec<LayerIdents> = (0..composed_shape.depth())
        .map(|i| tuple_layer_idents(field_idx, i))
        .collect();
    let layer_counters = tuple_layer_counters(field_idx, composed_shape.depth());

    let dummy_access = TokenStream::new();
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &dummy_access,
            idx: field_idx,
            name: column_prefix,
        },
        decimal128_encode_trait: &config.traits.decimal128_encode,
        paths: &config.external_paths,
    };
    let pep = super::super::vec::pep_for_primitive_leaf(leaf, &leaf_ctx, composed_shape);
    let leaf_projection_access = deepest_leaf_projection_access(composed_shape, projection, elem);

    let emitter = ShapeEmitter {
        shape: composed_shape,
        access: parent_access,
        layers: &layers,
        outer_some_prefix: idents::TUPLE_OUTER_SOME_PREFIX,
        precount_outer_some_prefix: idents::TUPLE_PRE_OUTER_SOME_PREFIX,
        total_counter: &total_leaves,
        layer_counters: &layer_counters,
        pp,
        pa_root,
        projection: projection.as_layer_projection(composed_shape),
    };
    let precount = emitter.precount();
    let leaf_body = tuple_scan_leaf_body(
        composed_shape,
        projection,
        leaf_projection_access.as_ref(),
        &pep.per_elem_push,
    );
    let scan = emitter.scan(&leaf_body, &pep.leaf_offsets_post_push);
    let counter_for_depth = |layer: usize| -> TokenStream {
        let counter = &layer_counters[layer];
        quote! { #counter }
    };
    let offsets_decls = emitter.offsets_decls(&counter_for_depth);
    let validity_decls = emitter.validity_decls(&counter_for_depth);

    let materialize = build_materialize(
        &emitter,
        &leaf_arr,
        &pep.leaf_logical_dtype,
        &pep.leaf_arr_expr,
        pp,
    );

    let extra_imports = pep.extra_imports;
    let storage_decls = pep.storage_decls;

    quote! {
        {
            let #series_local: #pp::Series = {
                #extra_imports
                #precount
                #storage_decls
                #offsets_decls
                #validity_decls
                #scan
                #materialize
            };
            let #named = #series_local.with_name(#column_prefix.into());
            columns.push(#named.into());
        }
    }
}

pub(super) fn tuple_layer_idents(field_idx: usize, layer: usize) -> LayerIdents {
    idents::LayerIds::new(idents::LayerNamespace::Tuple { field_idx }, layer).into()
}

pub(super) fn tuple_layer_counters(field_idx: usize, depth: usize) -> Vec<syn::Ident> {
    (0..depth.saturating_sub(1))
        .map(|layer| idents::tuple_layer_total(field_idx, layer))
        .collect()
}

pub(super) fn tuple_scan_leaf_body<'a>(
    shape: &'a VecLayers,
    projection: TupleProjection<'a>,
    projection_access: Option<&'a AccessChain>,
    per_elem_push: &'a TokenStream,
) -> impl Fn(&TokenStream) -> TokenStream + 'a {
    let leaf_bind = idents::leaf_value();
    move |vec_bind: &TokenStream| -> TokenStream {
        if let Some(element_access) = projection_access {
            return projected_leaf_body(vec_bind, projection, element_access, per_elem_push);
        }
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

fn build_materialize(
    emitter: &ShapeEmitter<'_>,
    leaf_arr: &syn::Ident,
    leaf_logical_dtype: &TokenStream,
    leaf_arr_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let pa_root = emitter.pa_root;
    let seed_arrow_dtype_id = idents::seed_arrow_dtype();
    let seed_dtype_decl = quote! {
        let #seed_arrow_dtype_id: #pa_root::datatypes::ArrowDataType =
            #pa_root::array::Array::dtype(&#leaf_arr).clone();
    };
    let wrap_layers = emitter.layer_wraps_move();
    let stack = shape_assemble_list_stack(
        quote! { ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef },
        quote! { #seed_arrow_dtype_id },
        &wrap_layers,
        leaf_logical_dtype.clone(),
        pp,
        pa_root,
        &idents::tuple_layer_list_arr,
    );
    quote! {
        #leaf_arr_expr
        #seed_dtype_decl
        #stack
    }
}
