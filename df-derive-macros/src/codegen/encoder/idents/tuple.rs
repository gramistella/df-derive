use quote::format_ident;
use syn::Ident;

pub(in crate::codegen) fn tuple_proj_param() -> Ident {
    format_ident!("__df_derive_t")
}

pub(in crate::codegen) fn tuple_nested_inner_v() -> Ident {
    format_ident!("__df_derive_inner_v")
}
