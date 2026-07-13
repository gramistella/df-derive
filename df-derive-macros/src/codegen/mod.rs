mod asserts;
mod bounds;
mod column_emit;
mod columnar_impl;
mod config;
mod encoder;
pub mod external_paths;
mod nested_names;
mod source_access;
mod support;
mod type_deps;
mod type_registry;

use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

pub use config::{MacroConfig, build_macro_config};

pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let support = support::generate_support(ir, config);
    let columnar_impl = columnar_impl::generate_columnar_impl(ir, config);
    let eager_asserts = asserts::generate_eager_asserts(
        ir,
        &config.traits.columnar,
        &config.traits.decimal128_encode,
    );

    // Keep helper names private while still emitting inherent impls for the
    // target type. The list assembly wrapper is emitted only for derives that
    // actually need `LargeListArray` stacking, and the nested validation
    // helpers are emitted only for derives whose columnar path calls nested
    // `Columnar::encode`.
    quote! {
        const _: () = {
            #eager_asserts

            #support

            #columnar_impl
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        AccessChain, ColumnIR, FieldSource, LeafShape, LeafSpec, NestedNamePolicy, NonEmpty,
        NumericKind, StructIR, TerminalLeafSpec, VecLayerSpec, VecLayers, WrapperShape,
    };
    use quote::{format_ident, quote};

    fn test_config() -> MacroConfig {
        let dataframe_mod = quote! { crate::dataframe };
        MacroConfig {
            traits: config::RuntimeTraitPaths {
                to_dataframe: syn::parse_quote!(crate::dataframe::ToDataFrame),
                columnar: syn::parse_quote!(crate::dataframe::Columnar),
                row_batch: syn::parse_quote!(crate::dataframe::RowBatch),
                decimal128_encode: syn::parse_quote!(crate::dataframe::Decimal128Encode),
            },
            external_paths: external_paths::default_runtime_paths(&dataframe_mod),
        }
    }

    fn assert_generated_impl_is_automatically_derived(ir: &StructIR) {
        let generated = generate_code(ir, &test_config()).to_string();
        let struct_name = ir.name.to_string();
        let columnar_impl = format!(
            "# [automatically_derived] impl crate :: dataframe :: Columnar for {struct_name}"
        );

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
        ColumnIR::field(
            name.to_owned(),
            field_source(name),
            terminal_leaf(LeafSpec::Numeric(NumericKind::U32)),
            wrapper_shape,
            NestedNamePolicy::Field,
        )
    }

    fn nested_column(name: &str, wrapper_shape: WrapperShape) -> ColumnIR {
        ColumnIR::field(
            name.to_owned(),
            field_source(name),
            terminal_leaf(LeafSpec::Struct(syn::parse_quote!(Inner))),
            wrapper_shape,
            NestedNamePolicy::Field,
        )
    }

    fn terminal_leaf(leaf: LeafSpec) -> TerminalLeafSpec {
        TerminalLeafSpec::new(leaf).expect("test leaf should be terminal")
    }

    fn depth_one_vec_shape() -> WrapperShape {
        WrapperShape::Vec(VecLayers {
            layers: NonEmpty::new(
                VecLayerSpec {
                    access: AccessChain::empty(),
                },
                Vec::new(),
            ),
            inner_access: AccessChain::empty(),
        })
    }

    #[test]
    fn generated_columnar_impls_are_automatically_derived() {
        let empty_ir = StructIR {
            name: format_ident!("EmptyRow"),
            generics: syn::Generics::default(),
            columns: Vec::new(),
        };
        assert_generated_impl_is_automatically_derived(&empty_ir);

        let non_empty_ir = StructIR {
            name: format_ident!("Row"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::bare()))],
        };
        assert_generated_impl_is_automatically_derived(&non_empty_ir);
    }

    #[test]
    fn derive_emits_exactly_one_runtime_primitive() {
        let ir = StructIR {
            name: format_ident!("Row"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::bare()))],
        };
        let generated = generate_code(&ir, &test_config()).to_string();

        assert_eq!(generated.matches("fn encode <").count(), 1, "{generated}");
        for removed_method in [
            "columnar_to_dataframe",
            "columnar_from_refs",
            "fn to_dataframe",
            "fn empty_dataframe",
            "fn schema",
        ] {
            assert!(!generated.contains(removed_method), "{generated}");
        }
        assert!(
            !generated.contains(
                "# [automatically_derived] impl crate :: dataframe :: ToDataFrame for Row"
            ),
            "{generated}",
        );
    }

    #[test]
    fn generated_frames_use_the_declared_batch_height() {
        let empty_ir = StructIR {
            name: format_ident!("EmptyRow"),
            generics: syn::Generics::default(),
            columns: Vec::new(),
        };
        let empty = generate_code(&empty_ir, &test_config()).to_string();
        assert!(
            empty.contains("DataFrame :: empty_with_height (rows . len ())"),
            "{empty}"
        );

        let non_empty_ir = StructIR {
            name: format_ident!("Row"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::bare()))],
        };
        let non_empty = generate_code(&non_empty_ir, &test_config()).to_string();
        let columns = encoder::idents::columns();
        let explicit_height_constructor = format!("DataFrame :: new (rows . len () , {columns})");
        assert!(
            non_empty.contains(&explicit_height_constructor),
            "{non_empty}",
        );

        for generated in [&empty, &non_empty] {
            assert!(!generated.contains("new_infer_height"), "{generated}");
            assert!(!generated.contains("_dummy"), "{generated}");
            assert!(!generated.contains("drop_in_place"), "{generated}");
        }
    }

    #[test]
    fn list_assembly_helper_is_emitted_only_for_vec_shapes() {
        let scalar_ir = StructIR {
            name: format_ident!("ScalarRow"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::bare()))],
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

        let scalar_ir = StructIR {
            name: format_ident!("ScalarRow"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::bare()))],
        };
        let scalar = generate_code(&scalar_ir, &test_config()).to_string();
        assert!(!scalar.contains(&validate_nested_frame), "{scalar}");

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

        let nested_ir = StructIR {
            name: format_ident!("NestedRow"),
            generics: syn::Generics::default(),
            columns: vec![nested_column(
                "inner",
                WrapperShape::Leaf(LeafShape::bare()),
            )],
        };
        let nested = generate_code(&nested_ir, &test_config()).to_string();
        assert!(nested.contains(&validate_nested_frame), "{nested}");
        assert!(
            nested.contains("let actual_columns = df . columns ()"),
            "{nested}",
        );
        assert!(
            !nested.contains("let actual_schema = df . schema ()"),
            "{nested}",
        );
        assert!(nested.contains(". columns () ["), "{nested}");
        assert!(!nested.contains(". column ("), "{nested}");

        let tuple_nested_ir = StructIR {
            name: format_ident!("TupleNestedRow"),
            generics: syn::Generics::default(),
            columns: vec![ColumnIR::field(
                "pair.field_0".to_owned(),
                field_source("pair"),
                terminal_leaf(LeafSpec::Struct(syn::parse_quote!(Inner))),
                WrapperShape::Leaf(LeafShape::bare()),
                NestedNamePolicy::Field,
            )],
        };
        let tuple_nested = generate_code(&tuple_nested_ir, &test_config()).to_string();
        assert!(
            tuple_nested.contains(&validate_nested_frame),
            "{tuple_nested}"
        );
        assert!(tuple_nested.contains(". columns () ["), "{tuple_nested}");
        assert!(!tuple_nested.contains(". column ("), "{tuple_nested}");
    }

    #[test]
    fn builder_only_columnar_impl_omits_empty_row_loop() {
        let vec_ir = StructIR {
            name: format_ident!("VecOnlyRow"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("ids", depth_one_vec_shape())],
        };
        let generated = generate_code(&vec_ir, &test_config()).to_string();
        let empty_loop = format!(
            "for {} in rows . iter () {{ }}",
            encoder::idents::populator_iter()
        );

        assert!(!generated.contains(&empty_loop), "{generated}");
    }
}
