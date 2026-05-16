use crate::ir::{
    AccessStep, FieldColumn, FieldSource, LeafShape, TerminalLeafRoute, TupleParentOptionColumn,
    TupleProjectionPath, TupleProjectionStep, TupleStaticColumn, WrapperShape,
};
use proc_macro2::TokenStream;
use quote::quote;

use super::encoder::access_chain_to_option_ref;

pub(in crate::codegen) fn apply_outer_smart_ptr_deref(
    mut expr: TokenStream,
    depth: usize,
) -> TokenStream {
    for _ in 0..depth {
        expr = quote! { (*(#expr)) };
    }
    expr
}

pub(in crate::codegen) fn field_source_access(
    field: &FieldSource,
    it_ident: &syn::Ident,
) -> TokenStream {
    let raw = field.field_index.map_or_else(
        || {
            let id = &field.name;
            quote! { #it_ident.#id }
        },
        |i| {
            let li = syn::Index::from(i);
            quote! { #it_ident.#li }
        },
    );
    apply_outer_smart_ptr_deref(raw, field.outer_smart_ptr_depth)
}

pub(in crate::codegen) fn field_column_access(
    column: &FieldColumn,
    it_ident: &syn::Ident,
) -> TokenStream {
    field_source_access(column.source(), it_ident)
}

pub(in crate::codegen) fn tuple_static_access(
    column: &TupleStaticColumn,
    it_ident: &syn::Ident,
) -> TokenStream {
    let root_access = field_source_access(column.root(), it_ident);
    apply_tuple_path(root_access, column.path())
}

pub(in crate::codegen) fn tuple_parent_option_access(
    column: &TupleParentOptionColumn,
    it_ident: &syn::Ident,
) -> TokenStream {
    let root_access = field_source_access(column.root(), it_ident);
    let collapsed = access_chain_to_option_ref(&root_access, column.parent_access());
    project_parent_option_tuple_column(
        &collapsed,
        column.path(),
        is_copy_parent_option_projection(column),
    )
}

pub(in crate::codegen) fn projection_step_suffix(step: TupleProjectionStep) -> TokenStream {
    let index = syn::Index::from(step.index);
    quote! { .#index }
}

pub(in crate::codegen) fn tuple_parent_option_some_receiver(
    column: &TupleParentOptionColumn,
) -> Option<crate::codegen::type_registry::PrimitiveExprReceiver> {
    if is_copy_parent_option_projection(column) {
        return None;
    }

    let WrapperShape::Leaf(LeafShape::Optional {
        option_layers,
        access,
    }) = column.wrapper_shape()
    else {
        return None;
    };

    if option_layers.get() == 1 && access.is_single_plain_option() {
        Some(crate::codegen::type_registry::PrimitiveExprReceiver::RefRef)
    } else {
        None
    }
}

fn apply_tuple_path(mut expr: TokenStream, path: &TupleProjectionPath) -> TokenStream {
    for step in path.iter() {
        let index = syn::Index::from(step.index);
        expr = quote! { (#expr).#index };
        expr = apply_outer_smart_ptr_deref(expr, step.outer_smart_ptr_depth);
    }
    expr
}

fn project_tuple_path_ref(tuple_ref: &TokenStream, path: &TupleProjectionPath) -> TokenStream {
    let mut projected = quote! { *(#tuple_ref) };
    for step in path.iter() {
        let index = syn::Index::from(step.index);
        projected = quote! { (#projected).#index };
        projected = apply_outer_smart_ptr_deref(projected, step.outer_smart_ptr_depth);
    }
    quote! { &(#projected) }
}

fn project_parent_option_tuple_column(
    collapsed_parent: &TokenStream,
    path: &TupleProjectionPath,
    copy_projection: bool,
) -> TokenStream {
    let param = super::encoder::idents::tuple_proj_param();
    let projected_ref = project_tuple_path_ref(&quote! { #param }, path);
    if copy_projection {
        quote! { ((#collapsed_parent).map(|#param| *(#projected_ref))) }
    } else {
        quote! { ((#collapsed_parent).map(|#param| #projected_ref)) }
    }
}

fn is_copy_parent_option_projection(column: &TupleParentOptionColumn) -> bool {
    let TerminalLeafRoute::Primitive(primitive) = column.leaf_spec().route() else {
        return false;
    };
    if !primitive.is_copy() {
        return false;
    }

    let WrapperShape::Leaf(LeafShape::Optional { access, .. }) = column.wrapper_shape() else {
        return false;
    };
    let Some((first, rest)) = access.steps.split_first() else {
        return false;
    };
    matches!(first, AccessStep::Option)
        && rest.iter().all(|step| matches!(step, AccessStep::Option))
}
