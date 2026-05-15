use quote::format_ident;
use syn::Ident;

pub(in crate::codegen) fn primitive_buf(idx: usize) -> Ident {
    format_ident!("__df_derive_buf_{}", idx)
}

pub(in crate::codegen) fn primitive_validity(idx: usize) -> Ident {
    format_ident!("__df_derive_val_{}", idx)
}

pub(in crate::codegen) fn primitive_row_idx(idx: usize) -> Ident {
    format_ident!("__df_derive_ri_{}", idx)
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

pub(in crate::codegen) fn leaf_arr() -> Ident {
    format_ident!("__df_derive_leaf_arr")
}

pub(in crate::codegen) fn total_leaves() -> Ident {
    format_ident!("__df_derive_total_leaves")
}

pub(in crate::codegen) fn bool_values() -> Ident {
    format_ident!("__df_derive_values")
}

pub(in crate::codegen) fn bool_validity() -> Ident {
    format_ident!("__df_derive_validity")
}

pub(in crate::codegen) fn bool_inner_offsets() -> Ident {
    format_ident!("__df_derive_inner_offsets")
}

pub(in crate::codegen) fn list_offset() -> Ident {
    format_ident!("__df_derive_offset")
}

pub(in crate::codegen) fn vec_flat() -> Ident {
    format_ident!("__df_derive_flat")
}

pub(in crate::codegen) fn vec_view_buf() -> Ident {
    format_ident!("__df_derive_view_buf")
}

pub(in crate::codegen) fn vec_leaf_idx() -> Ident {
    format_ident!("__df_derive_leaf_idx")
}

pub(in crate::codegen) fn bitmap_builder() -> Ident {
    format_ident!("__df_derive_b")
}

pub(in crate::codegen) fn bool_bare_offsets_buf() -> Ident {
    format_ident!("__df_derive_offsets_buf")
}

pub(in crate::codegen) fn bool_bare_list_arr() -> Ident {
    format_ident!("__df_derive_list_arr")
}
