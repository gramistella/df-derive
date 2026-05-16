use quote::format_ident;
use syn::Ident;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum LayerNamespace {
    Vec,
    Nested { field_idx: usize },
    Tuple { field_idx: usize },
}

pub(in crate::codegen) struct LayerIdents {
    pub offsets: Ident,
    pub offsets_buf: Ident,
    pub validity_mb: Ident,
    pub validity_bm: Ident,
    pub bind: Ident,
}

impl LayerIdents {
    pub(in crate::codegen) fn new(namespace: LayerNamespace, layer: usize) -> Self {
        match namespace {
            LayerNamespace::Vec => Self {
                offsets: vec_layer_offsets(layer),
                offsets_buf: vec_layer_offsets_buf(layer),
                validity_mb: vec_layer_validity(layer),
                validity_bm: vec_layer_validity_bm(layer),
                bind: vec_layer_bind(layer),
            },
            LayerNamespace::Nested { field_idx } => Self {
                offsets: nested_layer_offsets(field_idx, layer),
                offsets_buf: nested_layer_offsets_buf(field_idx, layer),
                validity_mb: nested_layer_validity_mb(field_idx, layer),
                validity_bm: nested_layer_validity_bm(field_idx, layer),
                bind: nested_layer_bind(field_idx, layer),
            },
            LayerNamespace::Tuple { field_idx } => Self {
                offsets: tuple_layer_offsets(field_idx, layer),
                offsets_buf: tuple_layer_offsets_buf(field_idx, layer),
                validity_mb: tuple_layer_validity_mb(field_idx, layer),
                validity_bm: tuple_layer_validity_bm(field_idx, layer),
                bind: tuple_layer_bind(field_idx, layer),
            },
        }
    }
}

pub(in crate::codegen) fn vec_layer_offsets(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_off_{}", layer)
}

pub(in crate::codegen) fn vec_layer_validity(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_val_{}", layer)
}

pub(in crate::codegen) fn vec_layer_bind(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_bind_{}", layer)
}

pub(in crate::codegen) fn vec_layer_offsets_buf(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_off_buf_{}", layer)
}

pub(in crate::codegen) fn vec_layer_validity_bm(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_val_bm_{}", layer)
}

pub(in crate::codegen) fn vec_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_list_arr_{}", layer)
}

pub(in crate::codegen) fn vec_layer_total(layer: usize) -> Ident {
    format_ident!("__df_derive_total_layer_{}", layer)
}

pub(in crate::codegen) const VEC_OUTER_SOME_PREFIX: &str = "__df_derive_some_";

pub(in crate::codegen) fn nested_layer_offsets(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_off_{}_{}", idx, layer)
}

pub(in crate::codegen) fn nested_layer_offsets_buf(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_off_buf_{}_{}", idx, layer)
}

pub(in crate::codegen) fn nested_layer_validity_mb(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_valmb_{}_{}", idx, layer)
}

pub(in crate::codegen) fn nested_layer_validity_bm(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_valbm_{}_{}", idx, layer)
}

pub(in crate::codegen) fn nested_layer_bind(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_bind_{}_{}", idx, layer)
}

pub(in crate::codegen) fn nested_layer_total(layer: usize) -> Ident {
    format_ident!("__df_derive_n_total_layer_{}", layer)
}

pub(in crate::codegen) const NESTED_OUTER_SOME_PREFIX: &str = "__df_derive_n_some_";
pub(in crate::codegen) const NESTED_PRE_OUTER_SOME_PREFIX: &str = "__df_derive_n_pre_some_";

pub(in crate::codegen) fn nested_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_n_arr_{}", layer)
}

pub(in crate::codegen) fn tuple_layer_offsets(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_off_{}_{}", field_idx, layer)
}

pub(in crate::codegen) fn tuple_layer_offsets_buf(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_off_buf_{}_{}", field_idx, layer)
}

pub(in crate::codegen) fn tuple_layer_validity_mb(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_valmb_{}_{}", field_idx, layer)
}

pub(in crate::codegen) fn tuple_layer_validity_bm(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_valbm_{}_{}", field_idx, layer)
}

pub(in crate::codegen) fn tuple_layer_bind(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_bind_{}_{}", field_idx, layer)
}

pub(in crate::codegen) fn tuple_layer_total(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_total_{}_{}", field_idx, layer)
}

pub(in crate::codegen) fn tuple_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_t_arr_{}", layer)
}

pub(in crate::codegen) const TUPLE_OUTER_SOME_PREFIX: &str = "__df_derive_t_some_";
pub(in crate::codegen) const TUPLE_PRE_OUTER_SOME_PREFIX: &str = "__df_derive_t_pre_some_";
