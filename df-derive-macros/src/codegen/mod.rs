mod bounds;
mod columnar_impl;
mod encoder;
pub mod external_paths;
mod helpers;
mod schema_nested;
mod strategy;
mod support;
mod trait_impl;
mod type_registry;

use crate::ir::StructIR;
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

pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let support = support::generate_support(ir, config);
    let trait_impl = trait_impl::generate_trait_impl(ir, config);
    let columnar_impl = columnar_impl::generate_columnar_impl(ir, config);
    let eager_asserts = helpers::generate_eager_asserts(
        ir,
        &config.to_dataframe_trait_path,
        &config.columnar_trait_path,
        &config.decimal128_encode_trait_path,
    );

    // Keep helper names private while still emitting inherent impls for the
    // target type. The list assembly wrapper is emitted only for derives that
    // actually need `LargeListArray` stacking, and the nested validation
    // helpers are emitted only for derives whose columnar path calls nested
    // `Columnar::columnar_from_refs`.
    quote! {
        const _: () = {
            #eager_asserts

            #support

            #trait_impl
            #columnar_impl
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        AccessChain, FieldIR, LeafShape, LeafSpec, NonEmpty, NumericKind, StructIR, TupleElement,
        VecLayerSpec, VecLayers, WrapperShape,
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
