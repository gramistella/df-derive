mod columnar_impl;
mod encoder;
pub mod external_paths;
mod helpers;
mod nested;
mod strategy;
mod trait_impl;
mod type_registry;

use crate::ir::{DisplayBase, LeafSpec, StringyBase, StructIR, TupleElement};
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};

/// Macro-wide configuration for generated code
#[allow(clippy::struct_field_names)]
pub struct MacroConfig {
    /// Fully-qualified path to the `ToDataFrame` trait (e.g., `df_derive::dataframe::ToDataFrame`)
    pub to_dataframe_trait_path: TokenStream,
    /// Fully-qualified path to the Columnar trait (e.g., `df_derive::dataframe::Columnar`)
    pub columnar_trait_path: TokenStream,
    /// Fully-qualified path to the `Decimal128Encode` trait used by Decimal
    /// fields to dispatch the value-to-i128-mantissa rescale through
    /// user-controlled code. Defaults to a sibling of the `ToDataFrame` trait
    /// (`<default-or-user-mod>::Decimal128Encode`); custom decimal backends
    /// override this with `#[df_derive(decimal128_encode = "...")]`.
    pub decimal128_encode_trait_path: TokenStream,
    /// External runtime dependency roots (`polars::prelude`,
    /// `polars_arrow`) used by generated code.
    pub external_paths: external_paths::ExternalPaths,
}

fn resolve_dataframe_mod_for_crate(name: &str, lib_crate_name: &str) -> Option<TokenStream> {
    match crate_name(name) {
        Ok(FoundCrate::Name(resolved)) => {
            let ident = format_ident!("{}", resolved);
            Some(quote! { ::#ident::dataframe })
        }
        Ok(FoundCrate::Itself) if is_expanding_lib_target(lib_crate_name) => {
            Some(quote! { crate::dataframe })
        }
        Ok(FoundCrate::Itself) => {
            let ident = format_ident!("{}", lib_crate_name);
            Some(quote! { ::#ident::dataframe })
        }
        Err(_) => None,
    }
}

fn is_expanding_lib_target(lib_crate_name: &str) -> bool {
    // `proc_macro_crate` reports `Itself` for every target in a package.
    // Only the library target has `crate::dataframe`; package examples,
    // benches, and integration tests need the path through the library crate.
    std::env::var("CARGO_CRATE_NAME").as_deref() == Ok(lib_crate_name)
}

pub fn resolve_default_dataframe_mod() -> TokenStream {
    // Default discovery order:
    // - `df-derive` facade (`df_derive::dataframe`, or `crate::dataframe` inside the facade)
    // - `df-derive-core` shared runtime (`df_derive_core::dataframe`)
    // - `paft-utils` direct runtime (`paft_utils::dataframe`)
    // - `paft` facade (`paft::dataframe`)
    // - local fallback (`crate::core::dataframe`)
    resolve_dataframe_mod_for_crate("df-derive", "df_derive")
        .or_else(|| resolve_dataframe_mod_for_crate("df-derive-core", "df_derive_core"))
        .or_else(|| resolve_dataframe_mod_for_crate("paft-utils", "paft_utils"))
        .or_else(|| resolve_dataframe_mod_for_crate("paft", "paft"))
        .unwrap_or_else(|| quote! { crate::core::dataframe })
}

#[allow(clippy::too_many_lines)]
pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let trait_impl = trait_impl::generate_trait_impl(ir, config);
    let columnar_impl = columnar_impl::generate_columnar_impl(ir, config);
    let eager_asserts = helpers::generate_eager_asserts(ir);
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let assemble_helper = encoder::idents::assemble_helper();
    let list_assembly = encoder::idents::list_assembly();
    let validate_nested_frame = encoder::idents::validate_nested_frame();
    let validate_nested_column_dtype = encoder::idents::validate_nested_column_dtype();

    // Keep helper names private while still emitting inherent impls for the
    // target type. The list assembly wrapper is the only generated site that
    // calls `Series::from_chunks_and_dtype_unchecked`.
    quote! {
        const _: () = {
            #eager_asserts

            struct #list_assembly {
                list_arr: #pp::LargeListArray,
                logical_dtype: #pp::DataType,
            }

            impl #list_assembly {
                #[inline(always)]
                #[allow(clippy::inline_always)]
                fn new(
                    list_arr: #pp::LargeListArray,
                    inner_logical_dtype: #pp::DataType,
                ) -> Self {
                    Self {
                        list_arr,
                        logical_dtype: #pp::DataType::List(
                            ::std::boxed::Box::new(inner_logical_dtype),
                        ),
                    }
                }

                #[inline(always)]
                #[allow(clippy::inline_always)]
                fn into_series(self) -> #pp::PolarsResult<#pp::Series> {
                    let expected_arrow_dtype: #pa_root::datatypes::ArrowDataType =
                        self.logical_dtype
                            .to_physical()
                            .to_arrow(#pp::CompatLevel::newest());
                    let actual_arrow_dtype = #pa_root::array::Array::dtype(&self.list_arr);
                    if actual_arrow_dtype != &expected_arrow_dtype {
                        return ::std::result::Result::Err(#pp::polars_err!(
                            ComputeError:
                            "df-derive: list assembly dtype mismatch: actual Arrow dtype {:?}, logical dtype {:?}",
                            actual_arrow_dtype,
                            self.logical_dtype,
                        ));
                    }
                    let Self {
                        list_arr,
                        logical_dtype,
                    } = self;
                    // SAFETY: `Self::new` is the generated list assembly
                    // boundary. Every caller reaches it through
                    // `encoder::shape_walk::shape_assemble_list_stack`,
                    // which builds `list_arr` from the leaf/nested physical
                    // Arrow dtype and the same logical dtype that schema
                    // generation emits. The release-mode check above
                    // compares the final Arrow list dtype against
                    // `logical_dtype.to_physical()`, covering logical
                    // wrappers such as Date, Datetime, Duration, Time,
                    // Decimal, and nested List envelopes. This matters for
                    // safe manual `ToDataFrame` / `Columnar` implementations:
                    // a bad schema can no longer violate the unchecked
                    // constructor's dtype invariant.
                    unsafe {
                        ::std::result::Result::Ok(#pp::Series::from_chunks_and_dtype_unchecked(
                            "".into(),
                            ::std::vec![
                                ::std::boxed::Box::new(list_arr) as #pp::ArrayRef,
                            ],
                            &logical_dtype,
                        ))
                    }
                }
            }

            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn #assemble_helper(
                list_arr: #pp::LargeListArray,
                inner_logical_dtype: #pp::DataType,
            ) -> #pp::PolarsResult<#pp::Series> {
                #list_assembly::new(list_arr, inner_logical_dtype).into_series()
            }

            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn #validate_nested_frame(
                df: &#pp::DataFrame,
                expected_height: usize,
                type_name: &str,
            ) -> #pp::PolarsResult<()> {
                let actual_height = df.height();
                if actual_height != expected_height {
                    return ::std::result::Result::Err(#pp::polars_err!(
                        ComputeError:
                        "df-derive: nested Columnar::columnar_from_refs for {} returned height {}, expected {}",
                        type_name,
                        actual_height,
                        expected_height,
                    ));
                }
                ::std::result::Result::Ok(())
            }

            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn #validate_nested_column_dtype(
                series: &#pp::Series,
                column_name: &str,
                declared_dtype: &#pp::DataType,
            ) -> #pp::PolarsResult<()> {
                let actual_dtype = series.dtype();
                if actual_dtype != declared_dtype {
                    return ::std::result::Result::Err(#pp::polars_err!(
                        ComputeError:
                        "df-derive: nested column `{}` dtype mismatch: actual dtype {:?}, declared schema dtype {:?}",
                        column_name,
                        actual_dtype,
                        declared_dtype,
                    ));
                }
                ::std::result::Result::Ok(())
            }

            #trait_impl
            #columnar_impl
        };
    }
}

#[derive(Default)]
struct GenericRequirements {
    nested_params: Vec<syn::Ident>,
    nested_types: Vec<syn::Type>,
    decimal_params: Vec<syn::Ident>,
    decimal_types: Vec<syn::Type>,
    as_ref_str: Vec<syn::Ident>,
    as_ref_str_types: Vec<syn::Type>,
    display_params: Vec<syn::Ident>,
    display_types: Vec<syn::Type>,
}

fn push_unique(out: &mut Vec<syn::Ident>, ident: &syn::Ident) {
    if !out.iter().any(|existing| existing == ident) {
        out.push(ident.clone());
    }
}

fn contains_ident(items: &[syn::Ident], ident: &syn::Ident) -> bool {
    items.iter().any(|item| item == ident)
}

fn push_unique_type(out: &mut Vec<syn::Type>, ty: &syn::Type) {
    let key = ty.to_token_stream().to_string();
    if !out
        .iter()
        .any(|existing| existing.to_token_stream().to_string() == key)
    {
        out.push(ty.clone());
    }
}

fn collect_tuple_element_requirements(elem: &TupleElement, reqs: &mut GenericRequirements) {
    collect_leaf_requirements(&elem.leaf_spec, reqs);
}

fn collect_leaf_requirements(leaf: &LeafSpec, reqs: &mut GenericRequirements) {
    match leaf {
        LeafSpec::Generic(ident) => {
            push_unique(&mut reqs.nested_params, ident);
        }
        LeafSpec::Struct(ty) => {
            push_unique_type(&mut reqs.nested_types, ty);
        }
        LeafSpec::AsStr(StringyBase::Generic(ident)) => {
            push_unique(&mut reqs.as_ref_str, ident);
        }
        LeafSpec::AsStr(StringyBase::Struct(ty)) => {
            push_unique_type(&mut reqs.as_ref_str_types, ty);
        }
        LeafSpec::AsString(DisplayBase::Generic(ident)) => {
            push_unique(&mut reqs.display_params, ident);
        }
        LeafSpec::AsString(DisplayBase::Struct(ty)) => {
            push_unique_type(&mut reqs.display_types, ty);
        }
        LeafSpec::Tuple(elements) => {
            for elem in elements {
                collect_tuple_element_requirements(elem, reqs);
            }
        }
        LeafSpec::Numeric(_)
        | LeafSpec::String
        | LeafSpec::Bool
        | LeafSpec::DateTime(_)
        | LeafSpec::NaiveDateTime(_)
        | LeafSpec::NaiveDate
        | LeafSpec::NaiveTime
        | LeafSpec::Duration { .. }
        | LeafSpec::Decimal { .. }
        | LeafSpec::AsString(_)
        | LeafSpec::AsStr(_)
        | LeafSpec::Binary => {}
    }
}

fn collect_generic_requirements(ir: &StructIR) -> GenericRequirements {
    let mut reqs = GenericRequirements::default();

    for field in &ir.fields {
        collect_leaf_requirements(&field.leaf_spec, &mut reqs);

        for ident in &field.decimal_generic_params {
            push_unique(&mut reqs.decimal_params, ident);
        }

        if let Some(ty) = &field.decimal_backend_ty {
            push_unique_type(&mut reqs.decimal_types, ty);
        }
    }

    reqs
}

/// Build `impl_generics`, `ty_generics`, and `where_clause` token streams
/// suitable for splicing into an `impl` header. Generic bounds are driven by
/// each parameter's role: direct generic dataframe payloads need
/// `ToDataFrame` + `Columnar`, decimal backends need `Decimal128Encode`, generic `as_str`
/// leaves need `AsRef<str>`, generic `as_string` leaves need `Display`, and
/// concrete conversion/nested types receive exact `where` predicates.
pub fn impl_parts_with_bounds(
    ir: &StructIR,
    config: &MacroConfig,
) -> (TokenStream, TokenStream, TokenStream) {
    let mut generics = ir.generics.clone();
    let reqs = collect_generic_requirements(ir);

    let to_df_trait = &config.to_dataframe_trait_path;
    let columnar_trait = &config.columnar_trait_path;
    let decimal_trait = &config.decimal128_encode_trait_path;
    let to_df_bound: syn::TypeParamBound =
        syn::parse2(quote! { #to_df_trait }).expect("trait path should parse as bound");
    let columnar_bound: syn::TypeParamBound =
        syn::parse2(quote! { #columnar_trait }).expect("trait path should parse as bound");
    let decimal_bound: syn::TypeParamBound =
        syn::parse2(quote! { #decimal_trait }).expect("trait path should parse as bound");
    let as_ref_str_bound: syn::TypeParamBound = syn::parse2(quote! { ::core::convert::AsRef<str> })
        .expect("AsRef<str> should parse as bound");
    let display_bound: syn::TypeParamBound =
        syn::parse2(quote! { ::core::fmt::Display }).expect("Display should parse as bound");
    // No `Clone` bound: bulk emitters collect `Vec<&T>` and route through
    // `Columnar::columnar_from_refs`, and every primitive-vec branch in the
    // encoder IR borrows from the for-loop binding directly. A user with a
    // non-`Clone` payload (e.g. `T: ToDataFrame + Columnar` only) can derive
    // `ToDataFrame` on a struct holding `T` without that bound leaking from
    // the macro.
    for tp in generics.type_params_mut() {
        if contains_ident(&reqs.nested_params, &tp.ident) {
            tp.bounds.push(to_df_bound.clone());
            tp.bounds.push(columnar_bound.clone());
        }

        if contains_ident(&reqs.decimal_params, &tp.ident) {
            tp.bounds.push(decimal_bound.clone());
        }

        if contains_ident(&reqs.as_ref_str, &tp.ident) {
            tp.bounds.push(as_ref_str_bound.clone());
        }

        if contains_ident(&reqs.display_params, &tp.ident) {
            tp.bounds.push(display_bound.clone());
        }
    }

    if !reqs.nested_types.is_empty()
        || !reqs.as_ref_str_types.is_empty()
        || !reqs.display_types.is_empty()
        || !reqs.decimal_types.is_empty()
    {
        let where_clause_mut = generics.make_where_clause();

        for ty in &reqs.nested_types {
            let nested_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #nested_ty: #to_df_trait })
                    .expect("nested ToDataFrame where predicate should parse"),
            );
            where_clause_mut.predicates.push(
                syn::parse2(quote! { #nested_ty: #columnar_trait })
                    .expect("nested Columnar where predicate should parse"),
            );
        }
        for ty in &reqs.as_ref_str_types {
            let as_str_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #as_str_ty: ::core::convert::AsRef<str> })
                    .expect("as_str type where predicate should parse"),
            );
        }
        for ty in &reqs.display_types {
            let display_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #display_ty: ::core::fmt::Display })
                    .expect("as_string type where predicate should parse"),
            );
        }
        for ty in &reqs.decimal_types {
            let decimal_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #decimal_ty: #decimal_trait })
                    .expect("decimal backend where predicate should parse"),
            );
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    (
        quote! { #impl_generics },
        quote! { #ty_generics },
        quote! { #where_clause },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AccessChain, FieldIR, LeafSpec, NumericKind, StructIR, WrapperShape};
    use quote::{format_ident, quote};

    fn test_config() -> MacroConfig {
        let dataframe_mod = quote! { crate::dataframe };
        MacroConfig {
            to_dataframe_trait_path: quote! { crate::dataframe::ToDataFrame },
            columnar_trait_path: quote! { crate::dataframe::Columnar },
            decimal128_encode_trait_path: quote! { crate::dataframe::Decimal128Encode },
            external_paths: external_paths::default_runtime_paths(&dataframe_mod),
        }
    }

    fn assert_generated_impls_are_automatically_derived(ir: &StructIR) {
        let generated = generate_code(ir, &test_config()).to_string();
        let struct_name = ir.name.to_string();
        let to_df_impl = format!(
            "# [automatically_derived] impl crate :: dataframe :: ToDataFrame for {struct_name}"
        );
        let columnar_impl = format!(
            "# [automatically_derived] impl crate :: dataframe :: Columnar for {struct_name}"
        );

        assert!(generated.contains(&to_df_impl), "{generated}");
        assert!(generated.contains(&columnar_impl), "{generated}");
    }

    #[test]
    fn generated_trait_impls_are_automatically_derived() {
        let empty_ir = StructIR {
            name: format_ident!("EmptyRow"),
            generics: syn::Generics::default(),
            fields: Vec::new(),
        };
        assert_generated_impls_are_automatically_derived(&empty_ir);

        let non_empty_ir = StructIR {
            name: format_ident!("Row"),
            generics: syn::Generics::default(),
            fields: vec![FieldIR {
                name: format_ident!("id"),
                field_index: None,
                leaf_spec: LeafSpec::Numeric(NumericKind::U32),
                wrapper_shape: WrapperShape::Leaf {
                    option_layers: 0,
                    access: AccessChain::empty(),
                },
                decimal_generic_params: Vec::new(),
                decimal_backend_ty: None,
                outer_smart_ptr_depth: 0,
            }],
        };
        assert_generated_impls_are_automatically_derived(&non_empty_ir);
    }
}
