mod asserts;
mod bounds;
mod column_emit;
mod columnar_spec_impl;
mod config;
mod encoder;
pub mod external_paths;
mod nested_names;
mod schema;
mod schema_nested;
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
    let columnar_spec_impl = columnar_spec_impl::generate_columnar_spec_impl(ir, config);
    let eager_asserts = asserts::generate_eager_asserts(
        ir,
        &config.runtime.columnar,
        &config.runtime.decimal128_encode,
    );

    // Keep helper names private while still emitting inherent impls for the
    // target type. The list assembly wrapper is emitted only for derives that
    // actually need `LargeListArray` stacking.
    quote! {
        const _: () = {
            #eager_asserts

            #support

            #columnar_spec_impl
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
            runtime: config::RuntimeSurfacePaths {
                columnar: syn::parse_quote!(crate::dataframe::Columnar),
                columnar_spec: syn::parse_quote!(crate::dataframe::ColumnarSpec),
                column_sink: syn::parse_quote!(crate::dataframe::ColumnSink),
                decimal128_encode: syn::parse_quote!(crate::dataframe::Decimal128Encode),
            },
            external_paths: external_paths::default_runtime_paths(&dataframe_mod),
        }
    }

    fn assert_generated_impl_is_automatically_derived(ir: &StructIR) {
        let generated = generate_code(ir, &test_config()).to_string();
        let struct_name = ir.name.to_string();
        let columnar_spec_impl = format!(
            "# [automatically_derived] impl crate :: dataframe :: ColumnarSpec for {struct_name}"
        );

        assert!(generated.contains(&columnar_spec_impl), "{generated}");
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
    fn generated_columnar_specs_are_automatically_derived() {
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

        assert_eq!(
            generated.matches("fn build_schema").count(),
            1,
            "{generated}"
        );
        assert_eq!(
            generated.matches("fn encode_columns <").count(),
            1,
            "{generated}",
        );
        for removed_method in [
            "columnar_to_dataframe",
            "columnar_from_refs",
            "fn encode <",
            "fn to_dataframe",
            "fn empty_dataframe",
            "fn schema",
        ] {
            assert!(!generated.contains(removed_method), "{generated}");
        }
        assert!(
            !generated
                .contains("# [automatically_derived] impl crate :: dataframe :: Columnar for Row"),
            "{generated}",
        );
        assert!(
            !generated.contains(
                "# [automatically_derived] impl crate :: dataframe :: ToDataFrame for Row"
            ),
            "{generated}",
        );
    }

    #[test]
    fn generated_specs_emit_columns_without_constructing_frames() {
        let empty_ir = StructIR {
            name: format_ident!("EmptyRow"),
            generics: syn::Generics::default(),
            columns: Vec::new(),
        };
        let empty = generate_code(&empty_ir, &test_config()).to_string();
        assert!(!empty.contains("DataFrame"), "{empty}");
        assert!(empty.contains("Iterator :: collect"), "{empty}");

        let non_empty_ir = StructIR {
            name: format_ident!("Row"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::bare()))],
        };
        let non_empty = generate_code(&non_empty_ir, &test_config()).to_string();
        let sink = encoder::idents::column_sink_param(&syn::Generics::default());
        assert!(non_empty.contains(&format!("{sink} . push")), "{non_empty}");

        for generated in [&empty, &non_empty] {
            assert!(!generated.contains("DataFrame :: new"), "{generated}");
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
    fn nested_shapes_compose_checked_batches_without_child_frames() {
        let scalar_ir = StructIR {
            name: format_ident!("ScalarRow"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("id", WrapperShape::Leaf(LeafShape::bare()))],
        };
        let scalar = generate_code(&scalar_ir, &test_config()).to_string();
        assert!(!scalar.contains("encode_batch"), "{scalar}");

        let primitive_vec_ir = StructIR {
            name: format_ident!("PrimitiveVecRow"),
            generics: syn::Generics::default(),
            columns: vec![numeric_column("ids", depth_one_vec_shape())],
        };
        let primitive_vec = generate_code(&primitive_vec_ir, &test_config()).to_string();
        assert!(!primitive_vec.contains("encode_batch"), "{primitive_vec}");

        let nested_ir = StructIR {
            name: format_ident!("NestedRow"),
            generics: syn::Generics::default(),
            columns: vec![nested_column(
                "inner",
                WrapperShape::Leaf(LeafShape::bare()),
            )],
        };
        let nested = generate_code(&nested_ir, &test_config()).to_string();
        assert!(nested.contains("Columnar > :: encode_batch"), "{nested}");
        assert!(nested.contains(". into_columns ()"), "{nested}");
        assert!(
            nested.contains("ColumnarSpec > :: build_schema"),
            "{nested}"
        );
        assert!(!nested.contains("DataFrame"), "{nested}");
        assert!(!nested.contains("validate_nested_frame"), "{nested}");

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
            tuple_nested.contains("Columnar > :: encode_batch"),
            "{tuple_nested}"
        );
        assert!(tuple_nested.contains(". into_columns ()"), "{tuple_nested}");
        assert!(!tuple_nested.contains("DataFrame"), "{tuple_nested}");
    }

    #[test]
    fn builder_only_spec_omits_empty_row_loop() {
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

    #[test]
    fn list_columns_have_no_separate_precount_walk() {
        fn row_walks(input: syn::DeriveInput) -> usize {
            let ir = crate::parser::parse_to_ir(&input).expect("input should lower to IR");
            generate_code(&ir, &test_config())
                .to_string()
                .matches("in rows . iter () . copied ()")
                .count()
        }

        assert_eq!(
            row_walks(syn::parse_quote! {
                struct PrimitiveLists {
                    values: Vec<Vec<Option<u32>>>,
                }
            }),
            1,
        );
        assert_eq!(
            row_walks(syn::parse_quote! {
                struct BooleanLists {
                    values: Vec<bool>,
                }
            }),
            1,
        );
        assert_eq!(
            row_walks(syn::parse_quote! {
                struct NestedLists {
                    values: Vec<Inner>,
                }
            }),
            1,
        );
        assert_eq!(
            row_walks(syn::parse_quote! {
                struct ProjectedLists {
                    values: Vec<(u32, bool)>,
                }
            }),
            2,
            "each projected column should scan once until sibling traversal is grouped",
        );
    }
}
