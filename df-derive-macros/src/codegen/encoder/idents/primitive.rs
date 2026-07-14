use quote::format_ident;
use syn::Ident;

pub(in crate::codegen) fn primitive_buf(idx: usize) -> Ident {
    format_ident!("__df_derive_buf_{}", idx)
}

pub(in crate::codegen) fn primitive_validity(idx: usize) -> Ident {
    format_ident!("__df_derive_val_{}", idx)
}

pub(in crate::codegen) fn primitive_str_scratch(idx: usize) -> Ident {
    format_ident!("__df_derive_str_{}", idx)
}

pub(in crate::codegen) fn vec_field_series(idx: usize) -> Ident {
    format_ident!("__df_derive_field_series_{}", idx)
}

pub(in crate::codegen) fn multi_option_local(idx: usize) -> Ident {
    format_ident!("__df_derive_mo_{}", idx)
}

pub(in crate::codegen) fn leaf_value() -> Ident {
    format_ident!("__df_derive_v")
}

pub(in crate::codegen) fn leaf_value_raw() -> Ident {
    format_ident!("__df_derive_v_raw")
}

pub(in crate::codegen) fn leaf_reserve_len() -> Ident {
    format_ident!("__df_derive_leaf_reserve_len")
}

pub(in crate::codegen) fn leaf_arr() -> Ident {
    format_ident!("__df_derive_leaf_arr")
}

pub(in crate::codegen) fn bool_values(idx: usize) -> Ident {
    format_ident!("__df_derive_values_{}", idx)
}

pub(in crate::codegen) fn bool_validity(idx: usize) -> Ident {
    format_ident!("__df_derive_validity_{}", idx)
}

pub(in crate::codegen) fn list_offset() -> Ident {
    format_ident!("__df_derive_offset")
}

pub(in crate::codegen) fn vec_flat(idx: usize) -> Ident {
    format_ident!("__df_derive_flat_{}", idx)
}

pub(in crate::codegen) fn vec_view_buf(idx: usize) -> Ident {
    format_ident!("__df_derive_view_buf_{}", idx)
}
