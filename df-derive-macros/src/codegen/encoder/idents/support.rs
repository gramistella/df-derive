use quote::format_ident;
use syn::Ident;

pub(in crate::codegen) fn columns() -> Ident {
    format_ident!("columns")
}

pub(in crate::codegen) fn populator_iter() -> Ident {
    format_ident!("__df_derive_it")
}

pub(in crate::codegen) fn field_named_series() -> Ident {
    format_ident!("__df_derive_named")
}

pub(in crate::codegen) fn schema_wrapped_dtype() -> Ident {
    format_ident!("__df_derive_wrapped")
}

pub(in crate::codegen) fn assemble_helper() -> Ident {
    format_ident!("__df_derive_assemble_list_series_unchecked")
}

pub(in crate::codegen) fn list_assembly() -> Ident {
    format_ident!("__DfDeriveListAssembly")
}

pub(in crate::codegen) fn validate_nested_frame() -> Ident {
    format_ident!("__df_derive_validate_nested_frame")
}

pub(in crate::codegen) fn validate_nested_column_dtype() -> Ident {
    format_ident!("__df_derive_validate_nested_column_dtype")
}

pub(in crate::codegen) fn as_ref_str_assert_helper() -> Ident {
    format_ident!("__df_derive_assert_as_ref_str")
}

pub(in crate::codegen) fn display_assert_helper() -> Ident {
    format_ident!("__df_derive_assert_display")
}

pub(in crate::codegen) fn nested_traits_assert_helper() -> Ident {
    format_ident!("__df_derive_assert_nested_traits")
}

pub(in crate::codegen) fn decimal_backend_assert_helper() -> Ident {
    format_ident!("__df_derive_assert_decimal_backend")
}

pub(in crate::codegen) fn collapse_option_param() -> Ident {
    format_ident!("__df_derive_o")
}
