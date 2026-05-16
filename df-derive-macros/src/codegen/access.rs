use crate::ir::FieldIR;
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

pub(in crate::codegen) fn field_access(field: &FieldIR, it_ident: &syn::Ident) -> TokenStream {
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
