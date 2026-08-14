mod asserts;
mod bounds;
mod column_emit;
mod columnar_spec_impl;
mod config;
mod encode_plan;
mod encoder;
pub mod external_paths;
mod nested_names;
mod planner;
mod schema;
mod schema_nested;
mod source_access;
mod type_deps;
mod type_registry;

use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

pub use config::{MacroConfig, build_macro_config};

pub fn generate_code(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let columnar_spec_impl = columnar_spec_impl::generate_columnar_spec_impl(ir, config);
    let eager_asserts = asserts::generate_eager_asserts(
        ir,
        &config.runtime.columnar,
        &config.runtime.decimal128_encode,
    );

    quote! {
        const _: () = {
            #eager_asserts

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
                row_cursor: syn::parse_quote!(crate::dataframe::RowCursor),
                columnar_spec: syn::parse_quote!(crate::dataframe::ColumnarSpec),
                column_sink: syn::parse_quote!(crate::dataframe::ColumnSink),
                encode_support: syn::parse_quote!(crate::dataframe::__private::encode),
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

    fn encode_plan(input: &syn::DeriveInput) -> encode_plan::EncodePlan {
        let ir = parse_ir(input);
        let config = test_config();
        let row = encoder::idents::populator_iter();
        let row_capacity = encoder::idents::row_capacity(&ir.generics);
        let sink = encoder::idents::column_sink_param(&ir.generics);
        let replay = quote! { crate::dataframe::RowCursor::replay(&*rows) };
        let static_tuple_plan = planner::StaticTuplePlan::select(&ir);

        columnar_spec_impl::prepare_encode_plan(
            &ir,
            &config,
            &row,
            &replay,
            static_tuple_plan,
            &row_capacity,
            &sink,
        )
    }

    fn enable_replay_call() -> String {
        let row_capacity = encoder::idents::row_capacity(&syn::Generics::default());
        format!("crate :: dataframe :: RowCursor :: enable_replay (rows , {row_capacity})")
    }

    fn assert_replay_policy(generated: &str, requires_replay: bool) {
        let policy = format!("const REQUIRES_ROW_REPLAY : bool = {requires_replay}");
        assert_eq!(generated.matches(&policy).count(), 1, "{generated}");
        assert!(!generated.contains("record_for_replay"), "{generated}");
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
        assert!(!empty.contains("record_for_replay"), "{empty}");
        assert_eq!(empty.matches("in rows . by_ref ()").count(), 1, "{empty}");
        assert_replay_policy(&empty, false);

        let non_empty = generated(&syn::parse_quote! {
            struct Row {
                id: u32,
            }
        });
        let sink = encoder::idents::column_sink_param(&syn::Generics::default());
        assert!(
            non_empty.contains(&format!("{sink} . next_slot")),
            "{non_empty}"
        );
        assert!(non_empty.contains(". commit"), "{non_empty}");
        assert!(non_empty.contains(". name"), "{non_empty}");
        assert!(
            !non_empty.contains(&format!("{sink} . push")),
            "{non_empty}"
        );
        assert!(!non_empty.contains("record_for_replay"), "{non_empty}");
        assert_replay_policy(&non_empty, false);

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
        assert!(
            nested.contains("Columnar > :: encode_ref_batch"),
            "{nested}"
        );
        assert!(nested.contains(". as_slice ()"), "{nested}");
        assert!(!nested.contains(". iter () . copied ()"), "{nested}");
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
            tuple_nested.contains("Columnar > :: encode_ref_batch"),
            "{tuple_nested}"
        );
        assert!(tuple_nested.contains(". into_columns ()"), "{tuple_nested}");
        assert!(!tuple_nested.contains("DataFrame"), "{tuple_nested}");
    }

    #[test]
    fn generated_encoder_consumes_source_once_and_captures_primitive_segments() {
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
        let captured_segments = "crate :: dataframe :: __private :: encode :: CapturedSegments";
        let enable_replay = enable_replay_call();

        assert_eq!(mixed.matches("in rows . by_ref ()").count(), 1, "{mixed}");
        assert_eq!(mixed.matches("Iterator :: size_hint").count(), 1, "{mixed}");
        assert!(!mixed.contains("Iterator :: collect"), "{mixed}");
        assert!(!mixed.contains(&enable_replay), "{mixed}");
        assert!(mixed.contains(captured_segments), "{mixed}");
        assert_replay_policy(&mixed, false);
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
        let shallow_segments = encoder::idents::vec_leaf_segments(scope, 0).to_string();
        let deep_segments = encoder::idents::vec_leaf_segments(scope, 1).to_string();
        let deep_count = encoder::idents::vec_leaf_count(scope, 1).to_string();
        let exact_buffer = "crate :: dataframe :: __private :: encode :: ExactBuffer";
        let prepared_validity = "crate :: dataframe :: __private :: encode :: PreparedValidity";
        let prepared_boolean = "crate :: dataframe :: __private :: encode :: PreparedBooleanValues";
        let row_cursor = "crate :: dataframe :: RowCursor";
        let enable_replay = enable_replay_call();
        let replay = format!("{row_cursor} :: replay (& * rows)");

        assert_eq!(
            deferred.matches("in rows . by_ref ()").count(),
            1,
            "{deferred}",
        );
        assert!(!deferred.contains("Iterator :: collect"), "{deferred}");
        assert!(
            deferred.matches(prepared_validity).count() >= 2,
            "{deferred}",
        );
        assert!(deferred.contains(&shallow_segments), "{deferred}");
        assert!(deferred.contains(&deep_segments), "{deferred}");
        assert!(!deferred.contains(&enable_replay), "{deferred}");
        assert!(!deferred.contains(&replay), "{deferred}");
        assert_replay_policy(&deferred, false);
        assert!(deferred.contains(exact_buffer), "{deferred}");
        assert!(deferred.contains(prepared_boolean), "{deferred}");
        assert!(!deferred.contains("unsafe"), "{deferred}");
        assert!(!deferred.contains("as_mut_ptr"), "{deferred}");
        let derived_impl = deferred
            .find("# [automatically_derived] impl")
            .expect("generated ColumnarSpec impl");
        let deferred_impl = &deferred[derived_impl..];
        assert!(
            deferred_impl.contains(&format!("ExactBuffer :: with_exact_len ({deep_count})")),
            "{deferred}",
        );
        assert!(
            deferred_impl.contains(". extend_nullable_captured"),
            "{deferred}"
        );
        assert!(
            deferred_impl.contains(". extend_nullable_options_captured"),
            "{deferred}"
        );
        assert!(deferred_impl.contains(". push"), "{deferred}");
        assert!(!deferred_impl.contains("unsafe"), "{deferred}");

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
        assert!(!immediate.contains(&enable_replay), "{immediate}");
        assert!(!immediate.contains(&replay), "{immediate}");
        assert_replay_policy(&immediate, false);
    }

    #[test]
    fn nullable_numeric_lists_choose_segment_or_group_capture_by_depth() {
        let captured_segments = "crate :: dataframe :: __private :: encode :: CapturedSegments";
        let captured_groups = "crate :: dataframe :: __private :: encode :: CapturedSegmentGroups";
        let generics = syn::Generics::default();
        let scope = encoder::idents::GeneratedIdentScope::new(&generics);
        let leaf_segments = encoder::idents::vec_leaf_segments(scope, 0).to_string();
        let leaf_count = encoder::idents::vec_leaf_count(scope, 0).to_string();

        let shallow = generated(&syn::parse_quote! {
            struct ShallowNullableNumericList {
                values: Vec<Option<i32>>,
            }
        });
        assert!(shallow.contains(captured_segments), "{shallow}");
        assert!(!shallow.contains(captured_groups), "{shallow}");
        assert!(
            shallow.contains(". extend_nullable_options_captured"),
            "{shallow}"
        );
        assert!(!shallow.contains(". capture_group"), "{shallow}");
        assert!(
            !shallow.contains(". extend_nullable_options_grouped"),
            "{shallow}"
        );
        assert!(
            shallow.contains(&format!("ExactBuffer :: with_exact_len ({leaf_count})")),
            "{shallow}",
        );

        let deep = generated(&syn::parse_quote! {
            struct DeepNullableNumericList {
                values: Vec<Vec<Option<i32>>>,
            }
        });
        assert!(deep.contains(captured_groups), "{deep}");
        assert!(!deep.contains(captured_segments), "{deep}");
        assert!(deep.contains(". capture_group"), "{deep}");
        assert!(deep.contains(". extend_nullable_options_grouped"), "{deep}");
        assert!(
            !deep.contains(". extend_nullable_options_captured"),
            "{deep}"
        );
        assert!(
            deep.contains(&format!(
                "ExactBuffer :: with_exact_len ({leaf_segments} . len ())"
            )),
            "{deep}",
        );
    }

    #[test]
    fn typed_plan_derives_replay_from_actual_list_and_tuple_lowering() {
        let shallow_infallible = encode_plan(&syn::parse_quote! {
            struct ShallowInfallible {
                values: Vec<Option<bool>>,
            }
        });
        let deep_infallible = encode_plan(&syn::parse_quote! {
            struct DeepInfallible {
                values: Vec<Vec<Option<i32>>>,
            }
        });
        let fallible = encode_plan(&syn::parse_quote! {
            struct Fallible {
                #[df_derive(as_string)]
                values: Vec<DisplayValue>,
            }
        });
        let tuple_boundary_minus_one = encode_plan(&syn::parse_quote! {
            struct TupleBoundaryMinusOne {
                values: (
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                ),
            }
        });
        let tuple_boundary = encode_plan(&syn::parse_quote! {
            struct TupleBoundary {
                values: (
                    i64, i64, i64, i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64, i64, i64, i64,
                ),
            }
        });

        assert!(!shallow_infallible.requirements().requires_row_replay());
        assert!(!deep_infallible.requirements().requires_row_replay());
        assert_eq!(deep_infallible.row_replay_group_count(), 0);
        assert!(!fallible.requirements().requires_row_replay());
        assert!(
            !tuple_boundary_minus_one
                .requirements()
                .requires_row_replay()
        );
        assert!(tuple_boundary.requirements().requires_row_replay());
        assert_eq!(tuple_boundary.row_replay_group_count(), 2);
    }

    #[test]
    fn primitive_lists_reference_only_the_runtime_storage_their_plan_needs() {
        let exact_buffer = "crate :: dataframe :: __private :: encode :: ExactBuffer";
        let prepared_validity = "crate :: dataframe :: __private :: encode :: PreparedValidity";
        let prepared_boolean = "crate :: dataframe :: __private :: encode :: PreparedBooleanValues";
        let bulk_boolean = generated(&syn::parse_quote! {
            struct BulkBooleanList {
                values: Vec<Vec<bool>>,
            }
        });
        assert!(!bulk_boolean.contains(exact_buffer), "{bulk_boolean}");
        assert!(!bulk_boolean.contains(prepared_validity), "{bulk_boolean}");
        assert!(!bulk_boolean.contains(prepared_boolean), "{bulk_boolean}");

        let bitmap_only = generated(&syn::parse_quote! {
            struct ShallowBooleanList {
                values: Vec<bool>,
            }
        });
        assert!(!bitmap_only.contains(exact_buffer), "{bitmap_only}");
        assert!(!bitmap_only.contains(prepared_validity), "{bitmap_only}");
        assert!(bitmap_only.contains(prepared_boolean), "{bitmap_only}");

        let bare_numeric = generated(&syn::parse_quote! {
            struct BareNumericList {
                values: Vec<Vec<i32>>,
            }
        });
        assert!(bare_numeric.contains(exact_buffer), "{bare_numeric}");
        assert!(!bare_numeric.contains(prepared_validity), "{bare_numeric}");
        assert!(!bare_numeric.contains(prepared_boolean), "{bare_numeric}");

        let mapped_numeric = generated(&syn::parse_quote! {
            struct MappedNumericLists {
                nullable: Vec<Option<i32>>,
                #[df_derive(decimal(precision = 18, scale = 4))]
                fallible: Vec<DecimalValue>,
                #[df_derive(decimal(precision = 18, scale = 4))]
                nullable_fallible: Vec<Option<DecimalValue>>,
            }
        });
        assert!(mapped_numeric.contains(exact_buffer), "{mapped_numeric}");
        assert!(
            mapped_numeric.contains(prepared_validity),
            "{mapped_numeric}"
        );

        for generated in [&bulk_boolean, &bitmap_only, &bare_numeric, &mapped_numeric] {
            assert!(!generated.contains("unsafe fn"), "{generated}");
            assert!(!generated.contains("as_mut_ptr"), "{generated}");
            assert!(!generated.contains("set_unchecked"), "{generated}");
            assert!(!generated.contains("unsafe"), "{generated}");
        }
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
        let row_capacity = encoder::idents::row_capacity(&generics);
        let row_cursor = "crate :: dataframe :: RowCursor";
        let replay = format!("in {row_cursor} :: replay (& * rows)");
        let enable_replay = enable_replay_call();
        let yielded = format!("{row_cursor} :: yielded (& * rows)");

        assert!(!narrow.contains(&enable_replay), "{narrow}");
        assert!(!narrow.contains(&replay), "{narrow}");
        assert_eq!(narrow.matches("in rows . by_ref ()").count(), 1, "{narrow}");
        assert_replay_policy(&narrow, false);
        assert_eq!(wide.matches("in rows . by_ref ()").count(), 1, "{wide}");
        assert_eq!(
            wide.matches(&replay).count(),
            8,
            "each four-terminal sibling group must share one replay lane: {wide}",
        );
        assert_eq!(
            wide.matches(&enable_replay).count(),
            1,
            "the one-shot input must enable replay exactly once: {wide}",
        );
        assert!(
            wide.contains(&format!("let {row_capacity} : usize = {yielded}")),
            "replayed columns must allocate from the exact yielded length: {wide}",
        );
        assert!(!wide.contains("Iterator :: collect"), "{wide}");
        assert_replay_policy(&wide, true);

        assert_eq!(mixed.matches("in rows . by_ref ()").count(), 1, "{mixed}");
        assert_eq!(
            mixed.matches(&replay).count(),
            2,
            "the sixteen infallible mixed-tuple terminals should use two lanes: {mixed}",
        );
        assert_eq!(
            mixed.matches(&enable_replay).count(),
            1,
            "the mixed tuple must still enable source replay once: {mixed}",
        );
        assert!(
            mixed.contains("try_to_i128_mantissa"),
            "the Decimal terminal must remain on its fallible path: {mixed}",
        );
        assert_replay_policy(&mixed, true);
        assert_eq!(
            mixed_structural.matches(&replay).count(),
            2,
            "only the two row-aligned scalar sibling groups should replay: {mixed_structural}",
        );
        assert_eq!(
            mixed_structural.matches(&enable_replay).count(),
            1,
            "structural siblings must share one replay-enabled source: {mixed_structural}",
        );
        assert_eq!(
            mixed_structural.matches("in rows . by_ref ()").count(),
            1,
            "{mixed_structural}",
        );
        assert_replay_policy(&mixed_structural, true);
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

        let row_cursor = "crate :: dataframe :: RowCursor";
        let replay = format!("in {row_cursor} :: replay (& * rows)");
        let enable_replay = enable_replay_call();

        assert!(
            !boundary_minus_one.contains(&enable_replay),
            "fifteen tuple terminals must stay on the fused source pass: {boundary_minus_one}",
        );
        assert!(
            !boundary_minus_one.contains(&replay),
            "{boundary_minus_one}"
        );
        assert_eq!(
            boundary_minus_one.matches("in rows . by_ref ()").count(),
            1,
            "{boundary_minus_one}",
        );
        assert_replay_policy(&boundary_minus_one, false);
        assert_eq!(
            boundary.matches(&replay).count(),
            2,
            "sixteen consecutive terminals must replay in two eight-wide lanes: {boundary}",
        );
        assert_eq!(
            boundary.matches(&enable_replay).count(),
            1,
            "the boundary shape must enable source replay exactly once: {boundary}",
        );
        assert_eq!(
            boundary.matches("in rows . by_ref ()").count(),
            1,
            "{boundary}",
        );
        assert_replay_policy(&boundary, true);
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
