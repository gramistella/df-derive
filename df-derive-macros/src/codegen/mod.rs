mod asserts;
mod bounds;
mod column_emit;
mod columnar_impl;
mod config;
mod encoder;
pub mod external_paths;
mod schema;
mod schema_nested;
mod source_access;
mod support;
mod trait_impl;
mod type_deps;
mod type_registry;

use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

pub use config::{MacroConfig, build_macro_config};

pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let support = support::generate_support(ir, config);
    let trait_impl = trait_impl::generate_trait_impl(ir, config);
    let columnar_impl = columnar_impl::generate_columnar_impl(ir, config);
    let eager_asserts = asserts::generate_eager_asserts(
        ir,
        &config.traits.to_dataframe,
        &config.traits.columnar,
        &config.traits.decimal128_encode,
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
        AccessChain, ColumnIR, ColumnSource, FieldSource, LeafShape, LeafSpec, NonEmpty,
        NumericKind, StructIR, TerminalLeafSpec, VecLayerSpec, VecLayers, WrapperShape,
    };
    use quote::{format_ident, quote};

    fn test_config() -> MacroConfig {
        let dataframe_mod = quote! { crate::dataframe };
        MacroConfig {
            traits: config::RuntimeTraitPaths {
                to_dataframe: syn::parse_quote!(crate::dataframe::ToDataFrame),
                columnar: syn::parse_quote!(crate::dataframe::Columnar),
                decimal128_encode: syn::parse_quote!(crate::dataframe::Decimal128Encode),
            },
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

    fn field_source(name: &str) -> FieldSource {
        FieldSource {
            name: format_ident!("{}", name),
            field_index: None,
            outer_smart_ptr_depth: 0,
        }
    }

    fn numeric_column(name: &str, wrapper_shape: WrapperShape) -> ColumnIR {
        ColumnIR {
            name: name.to_owned(),
            source: ColumnSource::Field(field_source(name)),
            leaf_spec: terminal_leaf(LeafSpec::Numeric(NumericKind::U32)),
            wrapper_shape,
        }
    }

    fn nested_column(name: &str, wrapper_shape: WrapperShape) -> ColumnIR {
        ColumnIR {
            name: name.to_owned(),
            source: ColumnSource::Field(field_source(name)),
            leaf_spec: terminal_leaf(LeafSpec::Struct(syn::parse_quote!(Inner))),
            wrapper_shape,
        }
    }

    fn terminal_leaf(leaf: LeafSpec) -> TerminalLeafSpec {
        TerminalLeafSpec::new(leaf).expect("test leaf should be terminal")
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
            columns: Vec::new(),
        };
        assert_generated_impls_are_automatically_derived(&empty_ir);

        let non_empty_ir = StructIR {
            name: format_ident!("Row"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::Bare))],
        };
        assert_generated_impls_are_automatically_derived(&non_empty_ir);
    }

    #[test]
    fn list_assembly_helper_is_emitted_only_for_vec_shapes() {
        let scalar_ir = StructIR {
            name: format_ident!("ScalarRow"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::Bare))],
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
            columns: vec![numeric_column("ids", depth_one_vec_shape())],
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
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::Bare))],
        };
        let scalar = generate_code(&scalar_ir, &test_config()).to_string();
        assert!(!scalar.contains(&validate_nested_frame), "{scalar}");
        assert!(!scalar.contains(&validate_nested_column_dtype), "{scalar}");

        let primitive_vec_ir = StructIR {
            name: format_ident!("PrimitiveVecRow"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("ids", depth_one_vec_shape())],
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
            columns: vec![nested_column("inner", WrapperShape::Leaf(LeafShape::Bare))],
        };
        let nested = generate_code(&nested_ir, &test_config()).to_string();
        assert!(nested.contains(&validate_nested_frame), "{nested}");
        assert!(nested.contains(&validate_nested_column_dtype), "{nested}");

        let tuple_nested_ir = StructIR {
            name: format_ident!("TupleNestedRow"),
            generics: syn::Generics::default(),
            columns: vec![ColumnIR {
                name: "pair.field_0".to_owned(),
                source: ColumnSource::Field(field_source("pair")),
                leaf_spec: terminal_leaf(LeafSpec::Struct(syn::parse_quote!(Inner))),
                wrapper_shape: WrapperShape::Leaf(LeafShape::Bare),
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
            columns: vec![numeric_column("ids", depth_one_vec_shape())],
        };
        let generated = generate_code(&vec_ir, &test_config()).to_string();
        let empty_loop = format!("for {} in items {{ }}", encoder::idents::populator_iter());

        assert!(!generated.contains(&empty_loop), "{generated}");
    }
}
