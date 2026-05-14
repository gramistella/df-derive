mod bounds;
mod columnar_impl;
mod encoder;
pub mod external_paths;
mod helpers;
mod nested;
mod strategy;
mod trait_impl;
mod type_registry;

use crate::ir::{LeafSpec, StructIR, TupleElement};
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

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

fn leaf_needs_list_assembly(leaf: &LeafSpec) -> bool {
    match leaf {
        LeafSpec::Tuple(elements) => elements.iter().any(tuple_element_needs_list_assembly),
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
        | LeafSpec::Binary
        | LeafSpec::Struct(_)
        | LeafSpec::Generic(_) => false,
    }
}

fn tuple_element_needs_list_assembly(element: &TupleElement) -> bool {
    element.wrapper_shape.vec_depth() > 0 || leaf_needs_list_assembly(&element.leaf_spec)
}

fn needs_list_assembly(ir: &StructIR) -> bool {
    ir.fields.iter().any(|field| {
        field.wrapper_shape.vec_depth() > 0 || leaf_needs_list_assembly(&field.leaf_spec)
    })
}

fn leaf_needs_nested_validation(leaf: &LeafSpec) -> bool {
    match leaf {
        LeafSpec::Struct(_) | LeafSpec::Generic(_) => true,
        LeafSpec::Tuple(elements) => elements.iter().any(tuple_element_needs_nested_validation),
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
        | LeafSpec::Binary => false,
    }
}

fn tuple_element_needs_nested_validation(element: &TupleElement) -> bool {
    leaf_needs_nested_validation(&element.leaf_spec)
}

fn needs_nested_validation(ir: &StructIR) -> bool {
    ir.fields
        .iter()
        .any(|field| leaf_needs_nested_validation(&field.leaf_spec))
}

#[allow(clippy::too_many_lines)]
pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let trait_impl = trait_impl::generate_trait_impl(ir, config);
    let columnar_impl = columnar_impl::generate_columnar_impl(ir, config);
    let eager_asserts = helpers::generate_eager_asserts(
        ir,
        &config.to_dataframe_trait_path,
        &config.columnar_trait_path,
        &config.decimal128_encode_trait_path,
    );
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let assemble_helper = encoder::idents::assemble_helper();
    let list_assembly = encoder::idents::list_assembly();

    let list_assembly_helpers = if needs_list_assembly(ir) {
        quote! {
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
        }
    } else {
        TokenStream::new()
    };

    let nested_validation_helpers = if needs_nested_validation(ir) {
        let validate_nested_frame = encoder::idents::validate_nested_frame();
        let validate_nested_column_dtype = encoder::idents::validate_nested_column_dtype();

        quote! {
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
        }
    } else {
        TokenStream::new()
    };

    // Keep helper names private while still emitting inherent impls for the
    // target type. The list assembly wrapper is emitted only for derives that
    // actually need `LargeListArray` stacking, and the nested validation
    // helpers are emitted only for derives whose columnar path calls nested
    // `Columnar::columnar_from_refs`.
    quote! {
        const _: () = {
            #eager_asserts

            #list_assembly_helpers

            #nested_validation_helpers

            #trait_impl
            #columnar_impl
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        AccessChain, FieldIR, LeafShape, LeafSpec, NonEmpty, NumericKind, StructIR, VecLayerSpec,
        VecLayers, WrapperShape,
    };
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

    fn numeric_field(name: &str, wrapper_shape: WrapperShape) -> FieldIR {
        FieldIR {
            name: format_ident!("{}", name),
            field_index: None,
            leaf_spec: LeafSpec::Numeric(NumericKind::U32),
            wrapper_shape,
            decimal_generic_params: Vec::new(),
            decimal_backend_ty: None,
            outer_smart_ptr_depth: 0,
        }
    }

    fn nested_field(name: &str, wrapper_shape: WrapperShape) -> FieldIR {
        FieldIR {
            name: format_ident!("{}", name),
            field_index: None,
            leaf_spec: LeafSpec::Struct(syn::parse_quote!(Inner)),
            wrapper_shape,
            decimal_generic_params: Vec::new(),
            decimal_backend_ty: None,
            outer_smart_ptr_depth: 0,
        }
    }

    fn depth_one_vec_shape() -> WrapperShape {
        WrapperShape::Vec(VecLayers {
            layers: NonEmpty::new(
                VecLayerSpec {
                    option_layers_above: 0,
                    access: AccessChain::empty(),
                },
                Vec::new(),
            ),
            inner_option_layers: 0,
            inner_access: AccessChain::empty(),
        })
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
            fields: vec![numeric_field("id", WrapperShape::Leaf(LeafShape::Bare))],
        };
        assert_generated_impls_are_automatically_derived(&non_empty_ir);
    }

    #[test]
    fn list_assembly_helper_is_emitted_only_for_vec_shapes() {
        let scalar_ir = StructIR {
            name: format_ident!("ScalarRow"),
            generics: syn::Generics::default(),
            fields: vec![numeric_field("id", WrapperShape::Leaf(LeafShape::Bare))],
        };
        let scalar = generate_code(&scalar_ir, &test_config()).to_string();
        assert!(!scalar.contains("__DfDeriveListAssembly"), "{scalar}");
        assert!(
            !scalar.contains("from_chunks_and_dtype_unchecked"),
            "{scalar}"
        );
        assert!(!scalar.contains("unsafe"), "{scalar}");

        let vec_ir = StructIR {
            name: format_ident!("VecRow"),
            generics: syn::Generics::default(),
            fields: vec![numeric_field("ids", depth_one_vec_shape())],
        };
        let with_vec = generate_code(&vec_ir, &test_config()).to_string();
        assert!(with_vec.contains("__DfDeriveListAssembly"), "{with_vec}");
        assert!(
            with_vec.contains("from_chunks_and_dtype_unchecked"),
            "{with_vec}"
        );
        assert!(with_vec.contains("unsafe"), "{with_vec}");
    }

    #[test]
    fn nested_validation_helpers_are_emitted_only_for_nested_shapes() {
        let validate_nested_frame = encoder::idents::validate_nested_frame().to_string();
        let validate_nested_column_dtype =
            encoder::idents::validate_nested_column_dtype().to_string();

        let scalar_ir = StructIR {
            name: format_ident!("ScalarRow"),
            generics: syn::Generics::default(),
            fields: vec![numeric_field("id", WrapperShape::Leaf(LeafShape::Bare))],
        };
        let scalar = generate_code(&scalar_ir, &test_config()).to_string();
        assert!(!scalar.contains(&validate_nested_frame), "{scalar}");
        assert!(!scalar.contains(&validate_nested_column_dtype), "{scalar}");

        let primitive_vec_ir = StructIR {
            name: format_ident!("PrimitiveVecRow"),
            generics: syn::Generics::default(),
            fields: vec![numeric_field("ids", depth_one_vec_shape())],
        };
        let primitive_vec = generate_code(&primitive_vec_ir, &test_config()).to_string();
        assert!(
            !primitive_vec.contains(&validate_nested_frame),
            "{primitive_vec}"
        );
        assert!(
            !primitive_vec.contains(&validate_nested_column_dtype),
            "{primitive_vec}"
        );

        let nested_ir = StructIR {
            name: format_ident!("NestedRow"),
            generics: syn::Generics::default(),
            fields: vec![nested_field("inner", WrapperShape::Leaf(LeafShape::Bare))],
        };
        let nested = generate_code(&nested_ir, &test_config()).to_string();
        assert!(nested.contains(&validate_nested_frame), "{nested}");
        assert!(nested.contains(&validate_nested_column_dtype), "{nested}");

        let tuple_nested_ir = StructIR {
            name: format_ident!("TupleNestedRow"),
            generics: syn::Generics::default(),
            fields: vec![FieldIR {
                name: format_ident!("pair"),
                field_index: None,
                leaf_spec: LeafSpec::Tuple(vec![TupleElement {
                    leaf_spec: LeafSpec::Struct(syn::parse_quote!(Inner)),
                    wrapper_shape: WrapperShape::Leaf(LeafShape::Bare),
                    outer_smart_ptr_depth: 0,
                }]),
                wrapper_shape: WrapperShape::Leaf(LeafShape::Bare),
                decimal_generic_params: Vec::new(),
                decimal_backend_ty: None,
                outer_smart_ptr_depth: 0,
            }],
        };
        let tuple_nested = generate_code(&tuple_nested_ir, &test_config()).to_string();
        assert!(
            tuple_nested.contains(&validate_nested_frame),
            "{tuple_nested}"
        );
        assert!(
            tuple_nested.contains(&validate_nested_column_dtype),
            "{tuple_nested}"
        );
    }

    #[test]
    fn builder_only_columnar_impl_omits_empty_row_loop() {
        let vec_ir = StructIR {
            name: format_ident!("VecOnlyRow"),
            generics: syn::Generics::default(),
            fields: vec![numeric_field("ids", depth_one_vec_shape())],
        };
        let generated = generate_code(&vec_ir, &test_config()).to_string();
        let empty_loop = format!("for {} in items {{ }}", encoder::idents::populator_iter());

        assert!(!generated.contains(&empty_loop), "{generated}");
    }
}
