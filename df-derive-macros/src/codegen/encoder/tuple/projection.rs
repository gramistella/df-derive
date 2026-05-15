use crate::ir::{
    AccessChain, AccessStep, LeafShape, LeafSpec, TupleElement, VecLayers, WrapperShape,
};
use proc_macro2::TokenStream;
use quote::quote;

use super::super::idents;
use super::super::shape_walk::LayerProjection;
use super::super::{access_chain_to_option_ref, access_chain_to_ref};

fn is_copy_element_projection(elem: &TupleElement) -> bool {
    is_copy_leaf_for_projection(&elem.leaf_spec)
        && match &elem.wrapper_shape {
            WrapperShape::Leaf(LeafShape::Bare) => true,
            WrapperShape::Leaf(LeafShape::Optional { access, .. }) => access.is_only_options(),
            WrapperShape::Vec(_) => false,
        }
}

/// True when the element's leaf is `Copy` AND a primitive that the
/// standard Option-leaf encoder expects to receive by value (not by
/// reference). The parent-Option tuple projection first forms a smart-pointer
/// resolved element reference, so outer `Box<T>` / `&T` wrappers over these
/// leaves are still copy-projectable. Option-only element stacks over Copy
/// leaves are also copy-projectable because `Option<Copy>` is Copy. Vecs and
/// access chains with smart pointers below an Option stay reference-oriented.
const fn is_copy_leaf_for_projection(leaf: &LeafSpec) -> bool {
    matches!(
        leaf,
        LeafSpec::Numeric(_)
            | LeafSpec::Bool
            | LeafSpec::NaiveDate
            | LeafSpec::NaiveTime
            | LeafSpec::Duration { .. }
    )
}

pub(super) fn option_tuple_projection_receiver(
    elem: &TupleElement,
) -> Option<crate::codegen::type_registry::PrimitiveExprReceiver> {
    // Non-Copy tuple elements under a parent Option are projected as
    // `Option<&T>` to avoid a hidden `Clone` bound. The standard single-option
    // push matches through `&Option<_>`, so mapped leaves see `&&T` and must
    // dereference once before UFCS trait dispatch.
    if matches!(elem.wrapper_shape, WrapperShape::Leaf(LeafShape::Bare))
        && !is_copy_element_projection(elem)
    {
        Some(crate::codegen::type_registry::PrimitiveExprReceiver::RefRef)
    } else {
        None
    }
}

/// Compose the parent's collapsed Option (1 outer Option) with the element's
/// own wrapper shape. For a leaf-only element, stack one extra Option layer.
/// For a Vec-bearing element, attach the parent's Option as outer-list
/// validity on the element's outermost Vec layer.
pub(super) fn compose_option_with_element(elem_shape: &WrapperShape) -> WrapperShape {
    match elem_shape {
        WrapperShape::Leaf(LeafShape::Bare) => WrapperShape::Leaf(LeafShape::from_option_access(
            1,
            prepend_option_access(&AccessChain::empty()),
        )),
        WrapperShape::Leaf(LeafShape::Optional {
            option_layers,
            access,
        }) => WrapperShape::Leaf(LeafShape::from_option_access(
            1 + option_layers.get(),
            prepend_option_access(access),
        )),
        WrapperShape::Vec(layers) => {
            let mut new_layers = layers.layers.clone();
            new_layers[0].option_layers_above += 1;
            new_layers[0].access = prepend_option_access(&new_layers[0].access);
            WrapperShape::Vec(VecLayers {
                layers: new_layers,
                inner_option_layers: layers.inner_option_layers,
                inner_access: layers.inner_access.clone(),
            })
        }
    }
}

fn prepend_option_access(access: &AccessChain) -> AccessChain {
    let mut steps = Vec::with_capacity(access.steps.len() + 1);
    steps.push(AccessStep::Option);
    steps.extend(access.steps.iter().copied());
    AccessChain { steps }
}

pub(super) fn concat_access_chains(left: &AccessChain, right: &AccessChain) -> AccessChain {
    let mut steps = Vec::with_capacity(left.steps.len() + right.steps.len());
    steps.extend(left.steps.iter().copied());
    steps.extend(right.steps.iter().copied());
    AccessChain { steps }
}

pub(super) fn prepend_parent_option_access(
    parent_access: &AccessChain,
    access: &AccessChain,
) -> AccessChain {
    if parent_access.option_layers() > 0 {
        prepend_option_access(access)
    } else {
        access.clone()
    }
}

#[derive(Clone, Copy)]
pub(super) struct TupleProjection<'a> {
    pub(super) layer: usize,
    pub(super) path: &'a TokenStream,
    pub(super) parent_access: &'a AccessChain,
    pub(super) smart_ptr_depth: usize,
}

impl<'a> TupleProjection<'a> {
    pub(super) fn as_layer_projection(self, shape: &VecLayers) -> Option<LayerProjection<'a>> {
        (self.layer < shape.depth()).then_some(LayerProjection {
            layer: self.layer,
            path: self.path,
            parent_access: self.parent_access,
            smart_ptr_depth: self.smart_ptr_depth,
        })
    }
}

pub(super) fn deepest_leaf_projection_access(
    shape: &VecLayers,
    projection: TupleProjection<'_>,
    elem: &TupleElement,
) -> Option<AccessChain> {
    if projection.layer != shape.depth() {
        return None;
    }
    match &elem.wrapper_shape {
        WrapperShape::Leaf(LeafShape::Optional { access, .. }) => Some(access.clone()),
        WrapperShape::Leaf(LeafShape::Bare) => Some(AccessChain::empty()),
        WrapperShape::Vec(_) => None,
    }
}

fn project_tuple_element_ref_with_path(
    tuple_ref: &TokenStream,
    path: &TokenStream,
    smart_ptr_depth: usize,
) -> TokenStream {
    let mut projected = quote! { (*(#tuple_ref)) #path };
    for _ in 0..smart_ptr_depth {
        projected = quote! { (*(#projected)) };
    }
    quote! { &(#projected) }
}

fn project_tuple_element_ref(
    tuple_ref: &TokenStream,
    projection: TupleProjection<'_>,
) -> TokenStream {
    project_tuple_element_ref_with_path(tuple_ref, projection.path, projection.smart_ptr_depth)
}

pub(super) fn project_parent_option_tuple_element(
    collapsed_parent: &TokenStream,
    elem: &TupleElement,
    elem_idx: usize,
) -> TokenStream {
    let elem_li = syn::Index::from(elem_idx);
    let projection_path = quote! { .#elem_li };
    let proj_param = idents::tuple_proj_param();
    let projected_ref = project_tuple_element_ref_with_path(
        &quote! { #proj_param },
        &projection_path,
        elem.outer_smart_ptr_depth,
    );
    if is_copy_element_projection(elem) {
        quote! {
            ((#collapsed_parent).map(|#proj_param| *(#projected_ref)))
        }
    } else {
        quote! {
            ((#collapsed_parent).map(|#proj_param| #projected_ref))
        }
    }
}

fn apply_element_access(
    projected_ref: &TokenStream,
    element_access: &AccessChain,
) -> (TokenStream, bool) {
    if element_access.option_layers() > 0 {
        return (
            access_chain_to_option_ref(projected_ref, element_access),
            true,
        );
    }
    let chain_ref = access_chain_to_ref(projected_ref, element_access);
    (chain_ref.expr, chain_ref.has_option)
}

fn projected_leaf_expr(
    raw_bind: &syn::Ident,
    projection: TupleProjection<'_>,
    element_access: &AccessChain,
) -> (TokenStream, bool) {
    let raw_ref = quote! { #raw_bind };
    if projection.parent_access.option_layers() > 0 {
        let tuple_ref = access_chain_to_option_ref(&raw_ref, projection.parent_access);
        let param = idents::tuple_proj_param();
        let projected_ref = project_tuple_element_ref(&quote! { #param }, projection);
        if element_access.option_layers() > 0 {
            let elem_ref = access_chain_to_option_ref(&projected_ref, element_access);
            (quote! { (#tuple_ref).and_then(|#param| #elem_ref) }, true)
        } else {
            let elem_ref = access_chain_to_ref(&projected_ref, element_access).expr;
            (quote! { (#tuple_ref).map(|#param| #elem_ref) }, true)
        }
    } else {
        let tuple_ref = access_chain_to_ref(&raw_ref, projection.parent_access).expr;
        let projected_ref = project_tuple_element_ref(&tuple_ref, projection);
        apply_element_access(&projected_ref, element_access)
    }
}

pub(super) fn projected_leaf_body(
    vec_bind: &TokenStream,
    projection: TupleProjection<'_>,
    element_access: &AccessChain,
    per_elem_push: &TokenStream,
) -> TokenStream {
    let raw_bind = idents::leaf_value_raw();
    let leaf_bind = idents::leaf_value();
    let (leaf_expr, has_option) = projected_leaf_expr(&raw_bind, projection, element_access);
    if has_option {
        quote! {
            for #raw_bind in #vec_bind.iter() {
                let #leaf_bind: ::std::option::Option<_> = #leaf_expr;
                #per_elem_push
            }
        }
    } else {
        quote! {
            for #raw_bind in #vec_bind.iter() {
                let #leaf_bind = #leaf_expr;
                #per_elem_push
            }
        }
    }
}
