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
    fn list_assembly_uses_polars_checked_constructor() {
        let scalar = generated(&syn::parse_quote! {
            struct ScalarRow {
                id: u32,
            }
        });
        assert!(!scalar.contains("from_chunk_and_dtype"), "{scalar}");
        assert!(
            !scalar.contains("from_chunks_and_dtype_unchecked"),
            "{scalar}"
        );
        assert!(!scalar.contains("unsafe"), "{scalar}");

        let with_vec = generated(&syn::parse_quote! {
            struct VecRow {
                ids: Vec<String>,
            }
        });
        assert!(
            with_vec.contains("Series :: from_chunk_and_dtype"),
            "{with_vec}"
        );
        assert!(
            !with_vec.contains("from_chunks_and_dtype_unchecked"),
            "{with_vec}"
        );
        assert!(!with_vec.contains("unsafe"), "{with_vec}");
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
    fn generated_encoder_consumes_source_once_and_replays_selected_shapes() {
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
        let generics = syn::Generics::default();
        let scope = encoder::idents::GeneratedIdentScope::new(&generics);
        let replay_rows = encoder::idents::replay_rows(scope).to_string();
        let row = encoder::idents::populator_iter().to_string();
        let replay_push = format!("{replay_rows} . push ({row})");
        let replay_loop = format!("in {replay_rows} . iter () . copied ()");

        assert_eq!(mixed.matches("in rows . by_ref ()").count(), 1, "{mixed}");
        assert!(!mixed.contains("Iterator :: collect"), "{mixed}");
        assert_eq!(mixed.matches(&replay_push).count(), 1, "{mixed}");
        assert!(mixed.contains(&replay_loop), "{mixed}");
    }

    #[test]
    fn primitive_lists_select_deferred_or_immediate_leaf_fill_by_effects() {
        let deferred = generated(&syn::parse_quote! {
            struct InfallibleLists {
                booleans: Vec<Option<bool>>,
                integers: Option<Vec<Option<Vec<Option<i32>>>>>,
            }
        });
        let generics = syn::Generics::default();
        let scope = encoder::idents::GeneratedIdentScope::new(&generics);
        let replay_rows = encoder::idents::replay_rows(scope).to_string();
        let row = encoder::idents::populator_iter().to_string();
        let shallow_segments = encoder::idents::vec_leaf_segments(scope, 0).to_string();
        let deep_segments = encoder::idents::vec_leaf_segments(scope, 1).to_string();
        let deep_counts = encoder::idents::vec_shape_counts(scope, 1).to_string();
        let push_reserved = encoder::idents::push_reserved(scope).to_string();
        let set_prepared_bitmap = encoder::idents::set_prepared_bitmap(scope).to_string();
        let unsafe_push_reserved = format!("unsafe fn {push_reserved}");
        let unsafe_set_prepared_bitmap = format!("unsafe fn {set_prepared_bitmap}");
        let replay_push = format!("{replay_rows} . push ({row})");

        assert_eq!(
            deferred.matches("in rows . by_ref ()").count(),
            1,
            "{deferred}",
        );
        assert!(!deferred.contains("Iterator :: collect"), "{deferred}");
        assert!(
            deferred
                .matches("bitmap :: MutableBitmap :: from_len_set")
                .count()
                >= 2,
            "{deferred}",
        );
        assert_eq!(deferred.matches(". checked_add").count(), 4, "{deferred}");
        assert!(deferred.contains(&shallow_segments), "{deferred}");
        assert!(!deferred.contains(&deep_segments), "{deferred}");
        assert_eq!(deferred.matches(&replay_push).count(), 1, "{deferred}");
        assert!(deferred.contains(&unsafe_push_reserved), "{deferred}");
        assert!(deferred.contains(&unsafe_set_prepared_bitmap), "{deferred}");
        assert_eq!(deferred.matches("debug_assert !").count(), 2, "{deferred}");
        assert!(deferred.contains("set_prepared_bitmap"), "{deferred}");
        assert!(deferred.contains(". as_mut_ptr ()"), "{deferred}");
        let derived_impl = deferred
            .find("# [automatically_derived] impl")
            .expect("generated ColumnarSpec impl");
        let deferred_impl = &deferred[derived_impl..];
        assert_eq!(
            deferred_impl.matches(". checked_add").count(),
            4,
            "{deferred}"
        );
        assert!(
            deferred_impl.contains(&format!("Vec :: with_capacity ({deep_counts} [2usize])")),
            "{deferred}",
        );
        for layer in 0..2 {
            assert!(
                deferred_impl.contains(&format!(
                    "Vec :: with_capacity ({deep_counts} [{layer}usize] . saturating_add (1)"
                )),
                "{deferred}",
            );
            assert!(
                deferred_impl.contains(&format!(
                    "MutableBitmap :: with_capacity ({deep_counts} [{layer}usize])"
                )),
                "{deferred}",
            );
        }
        assert!(
            deferred_impl.matches("set_prepared_bitmap").count() >= 2,
            "{deferred}"
        );
        assert!(deferred_impl.contains("push_reserved"), "{deferred}");
        assert!(!deferred_impl.contains(". set ("), "{deferred}");
        assert!(deferred_impl.contains("unsafe"), "{deferred}");

        let immediate = generated(&syn::parse_quote! {
            struct FallibleList {
                #[df_derive(as_string)]
                values: Vec<DisplayValue>,
            }
        });
        assert_eq!(
            immediate.matches("in rows . by_ref ()").count(),
            1,
            "{immediate}",
        );
        let immediate_impl = &immediate[immediate
            .find("# [automatically_derived] impl")
            .expect("generated ColumnarSpec impl")..];
        assert!(!immediate_impl.contains(". checked_add"), "{immediate}");
        assert!(immediate_impl.contains(". reserve"), "{immediate}");
        assert!(!immediate.contains(&replay_rows), "{immediate}");
    }

    #[test]
    fn primitive_list_helpers_are_emitted_selectively() {
        let generics = syn::Generics::default();
        let scope = encoder::idents::GeneratedIdentScope::new(&generics);
        let push_reserved = encoder::idents::push_reserved(scope).to_string();
        let set_prepared_bitmap = encoder::idents::set_prepared_bitmap(scope).to_string();
        let unsafe_push_reserved = format!("unsafe fn {push_reserved}");
        let unsafe_set_prepared_bitmap = format!("unsafe fn {set_prepared_bitmap}");
        let bulk_boolean = generated(&syn::parse_quote! {
            struct BulkBooleanList {
                values: Vec<Vec<bool>>,
            }
        });
        assert!(!bulk_boolean.contains("push_reserved"), "{bulk_boolean}");
        assert!(
            !bulk_boolean.contains("set_prepared_bitmap"),
            "{bulk_boolean}",
        );

        let bitmap_only = generated(&syn::parse_quote! {
            struct ShallowBooleanList {
                values: Vec<bool>,
            }
        });
        assert!(!bitmap_only.contains("push_reserved"), "{bitmap_only}");
        assert!(bitmap_only.contains("set_prepared_bitmap"), "{bitmap_only}");

        let bare_numeric = generated(&syn::parse_quote! {
            struct BareNumericList {
                values: Vec<Vec<i32>>,
            }
        });
        assert!(
            bare_numeric.contains(&unsafe_push_reserved),
            "{bare_numeric}"
        );
        assert!(
            !bare_numeric.contains("set_prepared_bitmap"),
            "{bare_numeric}"
        );

        let mapped_numeric = generated(&syn::parse_quote! {
            struct MappedNumericLists {
                nullable: Vec<Option<i32>>,
                #[df_derive(decimal(precision = 18, scale = 4))]
                fallible: Vec<DecimalValue>,
                #[df_derive(decimal(precision = 18, scale = 4))]
                nullable_fallible: Vec<Option<DecimalValue>>,
            }
        });
        assert!(
            mapped_numeric.contains(&unsafe_push_reserved),
            "{mapped_numeric}"
        );
        assert!(
            mapped_numeric.contains(&unsafe_set_prepared_bitmap),
            "{mapped_numeric}",
        );
    }

    #[test]
    fn wide_static_tuples_replay_bounded_sibling_lanes() {
        let narrow = generated(&syn::parse_quote! {
            struct NarrowTuple {
                values: (i64, i64, i64, i64, i64, i64, i64, i64),
            }
        });
        let wide = generated(&syn::parse_quote! {
            struct WideTuples {
                t0: (i64, i64, i64, i64),
                t1: (i64, i64, i64, i64),
                t2: (i64, i64, i64, i64),
                t3: (i64, i64, i64, i64),
                t4: (i64, i64, i64, i64),
                t5: (i64, i64, i64, i64),
                t6: (i64, i64, i64, i64),
                t7: (Option<i64>, i64, i64, i64),
            }
        });
        let mixed = generated(&syn::parse_quote! {
            struct MixedWideTuple {
                values: (
                    Decimal,
                    i64, i64, i64, i64,
                    i64, i64, i64, i64,
                    i64, i64, i64, i64,
                    i64, i64, i64, i64,
                ),
            }
        });
        let mixed_structural = generated(&syn::parse_quote! {
            struct MixedStructuralTuple {
                values: (
                    (i64, i64, i64, i64, i64, i64, i64, i64),
                    Option<(i64, i64)>,
                    Vec<(i64, i64)>,
                    (i64, i64, i64, i64, i64, i64, i64, i64),
                ),
            }
        });

        let generics = syn::Generics::default();
        let ident_scope = encoder::idents::GeneratedIdentScope::new(&generics);
        let replay_rows = encoder::idents::replay_rows(ident_scope);
        let row_capacity = encoder::idents::row_capacity(&generics);
        let row = encoder::idents::populator_iter();

        assert!(!narrow.contains(&replay_rows.to_string()), "{narrow}");
        assert_eq!(wide.matches("in rows . by_ref ()").count(), 1, "{wide}");
        assert_eq!(
            wide.matches(&format!("in {replay_rows} . iter () . copied ()"))
                .count(),
            8,
            "each four-terminal sibling group must share one replay lane: {wide}",
        );
        assert_eq!(
            wide.matches(&format!("{replay_rows} . push ({row})"))
                .count(),
            1,
            "the one-shot input must be buffered exactly once: {wide}",
        );
        assert!(
            wide.contains(&format!(
                "let {row_capacity} : usize = {replay_rows} . len ()"
            )),
            "replayed columns must allocate from the exact buffered length: {wide}",
        );
        assert!(!wide.contains("Iterator :: collect"), "{wide}");

        assert_eq!(mixed.matches("in rows . by_ref ()").count(), 1, "{mixed}");
        assert_eq!(
            mixed
                .matches(&format!("in {replay_rows} . iter () . copied ()"))
                .count(),
            2,
            "the sixteen infallible mixed-tuple terminals should use two lanes: {mixed}",
        );
        assert_eq!(
            mixed
                .matches(&format!("{replay_rows} . push ({row})"))
                .count(),
            1,
            "the mixed tuple must still buffer the source once: {mixed}",
        );
        assert!(
            mixed.contains("try_to_i128_mantissa"),
            "the Decimal terminal must remain on its fallible path: {mixed}",
        );
        assert_eq!(
            mixed_structural
                .matches(&format!("in {replay_rows} . iter () . copied ()"))
                .count(),
            2,
            "only the two row-aligned scalar sibling groups should replay: {mixed_structural}",
        );
        assert_eq!(
            mixed_structural
                .matches(&format!("{replay_rows} . push ({row})"))
                .count(),
            1,
            "structural siblings must share the one source buffer: {mixed_structural}",
        );
    }

    #[test]
    fn static_tuple_replay_policy_boundary_and_lane_width_are_guarded() {
        let boundary_minus_one = generated(&syn::parse_quote! {
            struct TupleReplayBoundaryMinusOne {
                scalar: i64,
                values: (
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                ),
            }
        });
        let boundary = generated(&syn::parse_quote! {
            struct TupleReplayBoundary {
                values: (
                    i64, i64, i64, i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64, i64, i64, i64,
                ),
            }
        });

        let generics = syn::Generics::default();
        let ident_scope = encoder::idents::GeneratedIdentScope::new(&generics);
        let replay_rows = encoder::idents::replay_rows(ident_scope);
        let row = encoder::idents::populator_iter();

        assert!(
            !boundary_minus_one.contains(&replay_rows.to_string()),
            "fifteen tuple terminals must stay on the fused source pass: {boundary_minus_one}",
        );
        assert_eq!(
            boundary
                .matches(&format!("in {replay_rows} . iter () . copied ()"))
                .count(),
            2,
            "sixteen consecutive terminals must replay in two eight-wide lanes: {boundary}",
        );
        assert_eq!(
            boundary
                .matches(&format!("{replay_rows} . push ({row})"))
                .count(),
            1,
            "the boundary shape must buffer its source exactly once: {boundary}",
        );
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
