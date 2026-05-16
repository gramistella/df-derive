use crate::ir::{
    AccessStep, ColumnIR, ColumnSource, FieldSource, LeafShape, ProjectionContext,
    TerminalLeafRoute, TupleProjectionStep, WrapperShape,
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

pub(in crate::codegen) fn column_access(column: &ColumnIR, it_ident: &syn::Ident) -> TokenStream {
    match &column.source {
        ColumnSource::Field(field) => field_source_access(field, it_ident),
        ColumnSource::TupleProjection {
            root,
            path,
            context,
        } => {
            let root_access = field_source_access(root, it_ident);
            match context {
                ProjectionContext::Static => apply_tuple_path(root_access, path),
                ProjectionContext::ParentOption { access } => {
                    let collapsed = access_chain_to_option_ref(&root_access, access);
                    project_parent_option_tuple_column(
                        &collapsed,
                        path,
                        is_copy_parent_option_projection(column),
                    )
                }
                ProjectionContext::ParentVec { .. } => root_access,
            }
        }
    }
}

pub(in crate::codegen) fn projection_path_suffix(path: &[TupleProjectionStep]) -> TokenStream {
    let mut suffix = TokenStream::new();
    for step in path {
        let index = syn::Index::from(step.index);
        suffix.extend(quote! { .#index });
    }
    suffix
}

pub(in crate::codegen) fn column_option_some_receiver(
    column: &ColumnIR,
) -> Option<crate::codegen::type_registry::PrimitiveExprReceiver> {
    if !matches!(
        column.source,
        ColumnSource::TupleProjection {
            context: ProjectionContext::ParentOption { .. },
            ..
        }
    ) || is_copy_parent_option_projection(column)
    {
        return None;
    }

    let WrapperShape::Leaf(LeafShape::Optional {
        option_layers,
        access,
    }) = &column.wrapper_shape
    else {
        return None;
    };

    if option_layers.get() == 1 && access.is_single_plain_option() {
        Some(crate::codegen::type_registry::PrimitiveExprReceiver::RefRef)
    } else {
        None
    }
}

fn apply_tuple_path(mut expr: TokenStream, path: &[TupleProjectionStep]) -> TokenStream {
    for step in path {
        let index = syn::Index::from(step.index);
        expr = quote! { (#expr).#index };
        expr = apply_outer_smart_ptr_deref(expr, step.outer_smart_ptr_depth);
    }
    expr
}

fn project_tuple_path_ref(tuple_ref: &TokenStream, path: &[TupleProjectionStep]) -> TokenStream {
    let mut projected = quote! { *(#tuple_ref) };
    for step in path {
        let index = syn::Index::from(step.index);
        projected = quote! { (#projected).#index };
        projected = apply_outer_smart_ptr_deref(projected, step.outer_smart_ptr_depth);
    }
    quote! { &(#projected) }
}

fn project_parent_option_tuple_column(
    collapsed_parent: &TokenStream,
    path: &[TupleProjectionStep],
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

fn is_copy_parent_option_projection(column: &ColumnIR) -> bool {
    let TerminalLeafRoute::Primitive(primitive) = column.leaf_spec.route() else {
        return false;
    };
    if !primitive.is_copy() {
        return false;
    }

    let WrapperShape::Leaf(LeafShape::Optional { access, .. }) = &column.wrapper_shape else {
        return false;
    };
    let Some((first, rest)) = access.steps.split_first() else {
        return false;
    };
    matches!(first, AccessStep::Option)
        && rest.iter().all(|step| matches!(step, AccessStep::Option))
}
