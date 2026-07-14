use crate::ir::{FieldColumn, FieldSource};
use proc_macro2::TokenStream;
use quote::quote;

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
    row: &syn::Ident,
) -> TokenStream {
    let raw = field.field_index.map_or_else(
        || {
            let id = &field.name;
            quote! { #row.#id }
        },
        |index| {
            let index = syn::Index::from(index);
            quote! { #row.#index }
        },
    );
    apply_outer_smart_ptr_deref(raw, field.outer_smart_ptr_depth)
}

pub(in crate::codegen) fn field_column_access(
    column: &FieldColumn,
    row: &syn::Ident,
) -> TokenStream {
    field_source_access(column.source(), row)
}
