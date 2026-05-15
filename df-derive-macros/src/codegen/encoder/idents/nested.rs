use quote::format_ident;
use syn::Ident;

pub(in crate::codegen) fn nested_inner_chunk() -> Ident {
    format_ident!("__df_derive_inner_chunk")
}

pub(in crate::codegen) fn seed_arrow_dtype() -> Ident {
    format_ident!("__df_derive_seed_dt")
}

pub(in crate::codegen) fn nested_flat(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_flat_{}", idx)
}

pub(in crate::codegen) fn nested_positions(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_pos_{}", idx)
}

pub(in crate::codegen) fn nested_df(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_df_{}", idx)
}

pub(in crate::codegen) fn nested_take(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_take_{}", idx)
}

pub(in crate::codegen) fn nested_total(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_total_{}", idx)
}

pub(in crate::codegen) fn nested_prefixed_name() -> Ident {
    format_ident!("__df_derive_prefixed")
}

pub(in crate::codegen) fn nested_col_name() -> Ident {
    format_ident!("__df_derive_col_name")
}

pub(in crate::codegen) fn nested_col_dtype() -> Ident {
    format_ident!("__df_derive_dtype")
}

pub(in crate::codegen) fn nested_inner_series() -> Ident {
    format_ident!("__df_derive_inner")
}

pub(in crate::codegen) fn nested_inner_full() -> Ident {
    format_ident!("__df_derive_inner_full")
}

pub(in crate::codegen) fn nested_inner_col() -> Ident {
    format_ident!("__df_derive_inner_col")
}

pub(in crate::codegen) fn nested_inner_rech() -> Ident {
    format_ident!("__df_derive_inner_rech")
}

pub(in crate::codegen) fn nested_maybe() -> Ident {
    format_ident!("__df_derive_maybe")
}
