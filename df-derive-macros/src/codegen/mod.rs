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
    use quote::quote;

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

    fn parse_ir(input: &syn::DeriveInput) -> StructIR {
        crate::parser::parse_to_ir(input).expect("input should lower to IR")
    }

    fn generated(input: &syn::DeriveInput) -> String {
        generate_code(&parse_ir(input), &test_config()).to_string()
    }

    #[test]
    fn generated_columnar_specs_are_automatically_derived() {
        let empty_ir = parse_ir(&syn::parse_quote!(
            struct EmptyRow;
        ));
        assert_generated_impl_is_automatically_derived(&empty_ir);

        let non_empty_ir = parse_ir(&syn::parse_quote! {
            struct Row {
                id: u32,
            }
        });
        assert_generated_impl_is_automatically_derived(&non_empty_ir);
    }

    #[test]
    fn derive_emits_exactly_one_runtime_primitive() {
        let ir = parse_ir(&syn::parse_quote! {
            struct Row {
                id: u32,
            }
        });
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
        let empty = generated(&syn::parse_quote!(
            struct EmptyRow;
        ));
        assert!(!empty.contains("DataFrame"), "{empty}");
        assert!(!empty.contains("Iterator :: collect"), "{empty}");
        assert_eq!(empty.matches("in rows . by_ref ()").count(), 1, "{empty}");

        let non_empty = generated(&syn::parse_quote! {
            struct Row {
                id: u32,
            }
        });
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
        let scalar = generated(&syn::parse_quote! {
            struct ScalarRow {
                id: u32,
            }
        });
        assert!(!scalar.contains("__DfDeriveListAssembly"), "{scalar}");
        assert!(
            !scalar.contains("from_chunks_and_dtype_unchecked"),
            "{scalar}"
        );
        assert!(!scalar.contains("unsafe"), "{scalar}");

        let with_vec = generated(&syn::parse_quote! {
            struct VecRow {
                ids: Vec<u32>,
            }
        });
        assert!(with_vec.contains("__DfDeriveListAssembly"), "{with_vec}");
        assert!(
            with_vec.contains("from_chunks_and_dtype_unchecked"),
            "{with_vec}"
        );
        assert!(with_vec.contains("unsafe"), "{with_vec}");
    }

    #[test]
    fn nested_shapes_compose_checked_batches_without_child_frames() {
        let scalar = generated(&syn::parse_quote! {
            struct ScalarRow {
                id: u32,
            }
        });
        assert!(!scalar.contains("encode_batch"), "{scalar}");

        let primitive_vec = generated(&syn::parse_quote! {
            struct PrimitiveVecRow {
                ids: Vec<u32>,
            }
        });
        assert!(!primitive_vec.contains("encode_batch"), "{primitive_vec}");

        let nested = generated(&syn::parse_quote! {
            struct NestedRow {
                inner: Inner,
            }
        });
        assert!(nested.contains("Columnar > :: encode_batch"), "{nested}");
        assert!(nested.contains(". into_columns ()"), "{nested}");
        assert!(
            nested.contains("ColumnarSpec > :: build_schema"),
            "{nested}"
        );
        assert!(!nested.contains("DataFrame"), "{nested}");
        assert!(!nested.contains("validate_nested_frame"), "{nested}");

        let tuple_nested = generated(&syn::parse_quote! {
            struct TupleNestedRow {
                pair: (Inner, u32),
            }
        });
        assert!(
            tuple_nested.contains("Columnar > :: encode_batch"),
            "{tuple_nested}"
        );
        assert!(tuple_nested.contains(". into_columns ()"), "{tuple_nested}");
        assert!(!tuple_nested.contains("DataFrame"), "{tuple_nested}");
    }

    #[test]
    fn generated_encoder_consumes_every_shape_in_one_outer_loop() {
        let mixed = generated(&syn::parse_quote! {
            struct Mixed<T> {
                id: u32,
                name: Option<String>,
                values: Vec<Vec<Option<u32>>>,
                nested: Option<Vec<Inner>>,
                tuple: Option<Vec<(u32, Vec<bool>, Option<Inner>)>>,
                generic: T,
            }
        });

        assert_eq!(mixed.matches("in rows . by_ref ()").count(), 1, "{mixed}");
        assert!(!mixed.contains("Iterator :: collect"), "{mixed}");
        assert!(!mixed.contains("rows . iter"), "{mixed}");
    }

    #[test]
    fn tuple_siblings_share_their_source_projection() {
        let bare = generated(&syn::parse_quote! {
            struct TupleScalars {
                values: (u32, bool, String),
            }
        });
        let optional_nested = generated(&syn::parse_quote! {
            struct OptionalNestedTuple {
                values: Option<((u32, bool), String)>,
            }
        });
        let listed = generated(&syn::parse_quote! {
            struct TupleLists {
                values: Option<Vec<(u32, bool, String)>>,
            }
        });

        let generics = syn::Generics::default();
        let ident_scope = encoder::idents::GeneratedIdentScope::new(&generics);
        let row = encoder::idents::populator_iter();
        let tuple_item = encoder::idents::tuple_item(ident_scope, 0);
        let tuple_layer = encoder::idents::LayerIdents::tuple(ident_scope, 0, 0);

        for generated in [&bare, &optional_nested, &listed] {
            assert_eq!(
                generated.matches(&format!("{row} . values")).count(),
                1,
                "tuple siblings must bind their source once: {generated}",
            );
        }
        assert_eq!(
            listed.matches(&format!("for {tuple_item} in")).count(),
            1,
            "tuple siblings must share one item walk: {listed}",
        );
        assert_eq!(
            listed
                .matches(&format!("let mut {}", tuple_layer.offsets))
                .count(),
            1,
            "tuple siblings must share one offsets vector: {listed}",
        );
        assert_eq!(
            listed
                .matches(&format!("let mut {}", tuple_layer.validity_mb))
                .count(),
            1,
            "tuple siblings must share one validity bitmap: {listed}",
        );
    }
}
