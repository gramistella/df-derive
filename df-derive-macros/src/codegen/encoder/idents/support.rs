use proc_macro2::Span;
use quote::format_ident;
use syn::{GenericParam, Generics, Ident};

#[derive(Clone, Copy)]
pub(in crate::codegen) struct GeneratedIdentScope<'a> {
    generics: &'a Generics,
}

impl<'a> GeneratedIdentScope<'a> {
    pub(in crate::codegen) const fn new(generics: &'a Generics) -> Self {
        Self { generics }
    }

    pub(in crate::codegen) fn fresh(self, base: &str) -> Ident {
        fresh_generic_ident(self.generics, base)
    }
}

pub(in crate::codegen) fn column_sink_param(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_sink")
}

pub(in crate::codegen) fn schema_fields(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_schema_fields")
}

pub(in crate::codegen) fn schema_duplicate_name(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_duplicate_name")
}

pub(in crate::codegen) fn schema_cache(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_schema_cache")
}

pub(in crate::codegen) fn schema_built(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_schema_built")
}

pub(in crate::codegen) fn schema_nested_fields(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_nested_fields")
}

pub(in crate::codegen) fn schema_inner_name(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_inner_name")
}

pub(in crate::codegen) fn schema_inner_dtype(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_inner_dtype")
}

pub(in crate::codegen) fn schema_output_name(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_output_name")
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

pub(in crate::codegen) fn replay_rows(scope: GeneratedIdentScope<'_>) -> Ident {
    scope.fresh("__df_derive_replay_rows")
}

pub(in crate::codegen) fn row_capacity(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_row_capacity")
}

pub(in crate::codegen) fn row_upper_bound(scope: GeneratedIdentScope<'_>) -> Ident {
    scope.fresh("__df_derive_row_upper_bound")
}

pub(in crate::codegen) fn input_rows_exact(scope: GeneratedIdentScope<'_>) -> Ident {
    scope.fresh("__df_derive_input_rows_exact")
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

pub(in crate::codegen) fn field_output_series(scope: GeneratedIdentScope<'_>) -> Ident {
    scope.fresh("__df_derive_series")
}

pub(in crate::codegen) fn schema_wrapped_dtype(generics: &Generics) -> Ident {
    fresh_generic_ident(generics, "__df_derive_wrapped")
}

pub(in crate::codegen) fn assemble_helper() -> Ident {
    format_ident!("__df_derive_assemble_list_series_unchecked")
}

pub(in crate::codegen) fn list_assembly() -> Ident {
    format_ident!("__DfDeriveListAssembly")
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
            <
                '__df_derive_row,
                __DfDeriveRows,
                const __DfDeriveRows_1: usize,
                const rows: usize,
                const __df_derive_replay_rows: usize,
                const __df_derive_sink: usize,
                const __df_derive_row_upper_bound: usize,
                const __df_derive_input_rows_exact: usize,
            >
        );

        assert_eq!(row_iter_param(&generics), "__DfDeriveRows_2");
        assert_eq!(row_lifetime(&generics).to_string(), "'__df_derive_row_1");
        assert_eq!(rows_param(&generics), "rows_1");
        assert_eq!(
            replay_rows(GeneratedIdentScope::new(&generics)),
            "__df_derive_replay_rows_1"
        );
        assert_eq!(row_capacity(&generics), "__df_derive_row_capacity");
        assert_eq!(
            row_upper_bound(GeneratedIdentScope::new(&generics)),
            "__df_derive_row_upper_bound_1"
        );
        assert_eq!(
            input_rows_exact(GeneratedIdentScope::new(&generics)),
            "__df_derive_input_rows_exact_1"
        );
        assert_eq!(column_sink_param(&generics), "__df_derive_sink_1");
    }

    #[test]
    fn schema_locals_are_fresh_against_user_const_generics() {
        let generics: Generics = syn::parse_quote!(
            <
                const __df_derive_schema_fields: usize,
                const __df_derive_duplicate_name: usize,
                const __df_derive_schema_cache: usize,
                const __df_derive_schema_built: usize,
                const __df_derive_nested_fields: usize,
                const __df_derive_inner_name: usize,
                const __df_derive_inner_dtype: usize,
                const __df_derive_output_name: usize,
                const __df_derive_wrapped: usize,
            >
        );

        assert_eq!(schema_fields(&generics), "__df_derive_schema_fields_1");
        assert_eq!(
            schema_duplicate_name(&generics),
            "__df_derive_duplicate_name_1"
        );
        assert_eq!(schema_cache(&generics), "__df_derive_schema_cache_1");
        assert_eq!(schema_built(&generics), "__df_derive_schema_built_1");
        assert_eq!(
            schema_nested_fields(&generics),
            "__df_derive_nested_fields_1"
        );
        assert_eq!(schema_inner_name(&generics), "__df_derive_inner_name_1");
        assert_eq!(schema_inner_dtype(&generics), "__df_derive_inner_dtype_1");
        assert_eq!(schema_output_name(&generics), "__df_derive_output_name_1");
        assert_eq!(schema_wrapped_dtype(&generics), "__df_derive_wrapped_1");
    }
}
