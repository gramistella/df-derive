use syn::Ident;

use super::GeneratedIdentScope;

pub(in crate::codegen) fn tuple_proj_param(scope: GeneratedIdentScope<'_>) -> Ident {
    scope.fresh("__df_derive_t")
}

pub(in crate::codegen) fn tuple_item_count(
    scope: GeneratedIdentScope<'_>,
    group_idx: usize,
) -> Ident {
    scope.fresh(&format!("__df_derive_t_count_{group_idx}"))
}

pub(in crate::codegen) fn tuple_item(scope: GeneratedIdentScope<'_>, group_idx: usize) -> Ident {
    scope.fresh(&format!("__df_derive_t_item_{group_idx}"))
}

pub(in crate::codegen) fn tuple_value(scope: GeneratedIdentScope<'_>, group_idx: usize) -> Ident {
    scope.fresh(&format!("__df_derive_t_value_{group_idx}"))
}

pub(in crate::codegen) fn tuple_input(scope: GeneratedIdentScope<'_>, group_idx: usize) -> Ident {
    scope.fresh(&format!("__df_derive_t_input_{group_idx}"))
}

pub(in crate::codegen) fn tuple_prefix_inner_series(
    scope: GeneratedIdentScope<'_>,
    field_idx: usize,
) -> Ident {
    scope.fresh(&format!("__df_derive_t_inner_{field_idx}"))
}

pub(in crate::codegen) fn tuple_prefix_rechunked(
    scope: GeneratedIdentScope<'_>,
    field_idx: usize,
) -> Ident {
    scope.fresh(&format!("__df_derive_t_rech_{field_idx}"))
}

pub(in crate::codegen) fn tuple_prefix_chunk(
    scope: GeneratedIdentScope<'_>,
    field_idx: usize,
) -> Ident {
    scope.fresh(&format!("__df_derive_t_chunk_{field_idx}"))
}

pub(in crate::codegen) fn tuple_prefix_list_arr(
    scope: GeneratedIdentScope<'_>,
    field_idx: usize,
    layer: usize,
) -> Ident {
    scope.fresh(&format!("__df_derive_t_prefix_arr_{field_idx}_{layer}"))
}

pub(in crate::codegen) fn tuple_output_series(scope: GeneratedIdentScope<'_>) -> Ident {
    scope.fresh("__df_derive_tuple_series")
}

pub(in crate::codegen) fn tuple_logical_dtype(scope: GeneratedIdentScope<'_>) -> Ident {
    scope.fresh("__df_derive_t_logical_dtype")
}
