use proc_macro2::Span;
use quote::format_ident;
use syn::{GenericParam, Generics, Ident};

pub(in crate::codegen) fn columns() -> Ident {
    format_ident!("__df_derive_columns")
}

pub(in crate::codegen) fn populator_iter() -> Ident {
    format_ident!("__df_derive_it")
}

pub(in crate::codegen) fn row_iter_param(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__DfDeriveRows")
}

pub(in crate::codegen) fn row_lifetime(generics: &Generics) -> syn::Lifetime {
    let ident = fresh_generic_ident(generics, "__df_derive_row");
    syn::Lifetime::new(&format!("'{ident}"), Span::call_site())
}

pub(in crate::codegen) fn rows_param(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "rows")
}

fn fresh_generic_ident(generics: &Generics, base: &str) -> Ident {
    let mut suffix = 0_usize;
    loop {
        let candidate = if suffix == 0 {
            format_ident!("{base}")
        } else {
            format_ident!("{base}_{suffix}")
        };
        let is_used = generics.params.iter().any(|param| match param {
            GenericParam::Lifetime(param) => param.lifetime.ident == candidate,
            GenericParam::Type(param) => param.ident == candidate,
            GenericParam::Const(param) => param.ident == candidate,
        });
        if !is_used {
            return candidate;
        }
        suffix += 1;
    }
}

pub(in crate::codegen) fn field_named_series() -> Ident {
    format_ident!("__df_derive_named")
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

pub(in crate::codegen) fn validate_unique_column_names() -> Ident {
    format_ident!("__df_derive_validate_unique_column_names")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_parameters_are_fresh_against_user_generics() {
        let generics: Generics = syn::parse_quote!(
            <'__df_derive_row, __DfDeriveRows, const __DfDeriveRows_1: usize, const rows: usize>
        );

        assert_eq!(row_iter_param(&generics), "__DfDeriveRows_2");
        assert_eq!(row_lifetime(&generics).to_string(), "'__df_derive_row_1");
        assert_eq!(rows_param(&generics), "rows_1");
    }
}
