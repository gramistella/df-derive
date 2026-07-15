//! Encoding policy decisions shared by column lowering and support emission.
//!
//! This module chooses an execution shape; emitters remain responsible for
//! rendering that shape. Keeping the policy here prevents runtime replay,
//! helper generation, and hot-loop width from being inferred independently.

use crate::ir::{
    FieldPlan, PrimitiveLeaf, StructIR, TerminalLeafRoute, TupleField, TupleNode, TupleNodeKind,
    VecLayers, WrapperShape,
};

/// Whether a primitive list may revisit the caller's rows after the source
/// pass. Ordinary tuple-list lowering is deliberately unavailable because its
/// projected access may name locals owned by the fused scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum RowReplayCapability {
    Direct,
    Unavailable,
}

/// The sole primitive-list scheduling definition.
///
/// The unit specialization is the policy selected from IR. Lowering replaces
/// each unit with the distinct render payload required by that branch, so an
/// encoded plan cannot contain ingredients for a different branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum PrimitiveListPlan<Stream = (), Capture = (), Replay = (), Bulk = ()> {
    /// Evaluate the leaf in the source pass and grow reserved storage.
    StreamReserved(Stream),
    /// Capture stable innermost list references and fill after the source pass.
    CaptureSegments(Capture),
    /// Count the complete list shape, then revisit source rows for exact fill.
    ReplayRows(Replay),
    /// Append real innermost Boolean slices through a whole-segment API.
    BulkSegments(Bulk),
}

pub(in crate::codegen) type PrimitiveListPolicy = PrimitiveListPlan<(), (), (), ()>;

impl PrimitiveListPlan<(), (), (), ()> {
    pub(in crate::codegen) fn for_wrapper(
        leaf: PrimitiveLeaf<'_>,
        wrapper: &WrapperShape,
        replay: RowReplayCapability,
    ) -> Option<Self> {
        match wrapper {
            WrapperShape::Vec(shape) => Some(Self::select(leaf, shape, replay)),
            WrapperShape::Leaf(_) => None,
        }
    }

    pub(in crate::codegen) fn select(
        leaf: PrimitiveLeaf<'_>,
        shape: &VecLayers,
        replay: RowReplayCapability,
    ) -> Self {
        if matches!(leaf, PrimitiveLeaf::Bool)
            && !shape.has_inner_option()
            && shape.depth() >= 2
            && shape.inner_access.is_empty()
        {
            return Self::BulkSegments(());
        }
        if !leaf.evaluation_effect().allows_replay() {
            return Self::StreamReserved(());
        }
        if shape.depth() >= 2 && replay == RowReplayCapability::Direct {
            Self::ReplayRows(())
        } else {
            Self::CaptureSegments(())
        }
    }

    pub(in crate::codegen) const fn uses_exact_deferred_storage(self) -> bool {
        matches!(self, Self::CaptureSegments(()) | Self::ReplayRows(()))
    }
}

#[derive(Clone, Copy, Default)]
pub(in crate::codegen) struct SupportRequirements {
    pub push_reserved: bool,
    pub set_prepared_bitmap: bool,
}

impl SupportRequirements {
    fn include_primitive_list(
        &mut self,
        leaf: PrimitiveLeaf<'_>,
        shape: &VecLayers,
        replay: RowReplayCapability,
    ) {
        let plan = PrimitiveListPolicy::select(leaf, shape, replay);
        self.push_reserved |= is_fixed_width_primitive(leaf);
        self.set_prepared_bitmap |= shape.has_inner_option()
            || (matches!(leaf, PrimitiveLeaf::Bool) && plan.uses_exact_deferred_storage());
    }
}

const fn is_fixed_width_primitive(leaf: PrimitiveLeaf<'_>) -> bool {
    matches!(
        leaf,
        PrimitiveLeaf::Numeric(_)
            | PrimitiveLeaf::DateTime(_)
            | PrimitiveLeaf::NaiveDateTime(_)
            | PrimitiveLeaf::NaiveDate
            | PrimitiveLeaf::NaiveTime
            | PrimitiveLeaf::Duration { .. }
            | PrimitiveLeaf::Decimal { .. }
    )
}

pub(in crate::codegen) fn support_requirements(ir: &StructIR) -> SupportRequirements {
    fn include(
        needs: &mut SupportRequirements,
        leaf: PrimitiveLeaf<'_>,
        wrapper: &WrapperShape,
        replay: RowReplayCapability,
    ) {
        if let WrapperShape::Vec(shape) = wrapper {
            needs.include_primitive_list(leaf, shape, replay);
        }
    }

    fn visit_tuple(node: &TupleNode, needs: &mut SupportRequirements) {
        match node.kind() {
            TupleNodeKind::Leaf(common) => {
                if let TerminalLeafRoute::Primitive(leaf) = common.leaf_spec().route() {
                    include(
                        needs,
                        leaf,
                        node.wrapper_shape(),
                        RowReplayCapability::Unavailable,
                    );
                }
            }
            TupleNodeKind::Tuple(elements) => {
                for child in elements.iter() {
                    visit_tuple(child, needs);
                }
            }
        }
    }

    let mut needs = SupportRequirements::default();
    for field in &ir.fields {
        match field {
            FieldPlan::Column(column) => {
                if let TerminalLeafRoute::Primitive(leaf) = column.leaf_spec().route() {
                    include(
                        &mut needs,
                        leaf,
                        column.wrapper_shape(),
                        RowReplayCapability::Direct,
                    );
                }
            }
            FieldPlan::Tuple(tuple) => {
                for node in tuple.elements().iter() {
                    visit_tuple(node, &mut needs);
                }
            }
        }
    }
    needs
}

// Below this width, one fused source pass is cheaper than setting up and
// revisiting the runtime row cursor once per safe scalar column. At and above
// it, one giant row-wise push loop creates enough live buffer state to lose
// decisively to narrow column-at-a-time loops. The boundary benchmarks own
// this empirical policy.
const STATIC_TUPLE_REPLAY_MIN_TERMINALS: usize = 16;
// Eight terminals keeps each replay hot loop narrow. Structural boundaries in
// tuple lowering still flush lanes early; this is only the maximum lane width.
pub(in crate::codegen) const STATIC_TUPLE_REPLAY_LANE_WIDTH: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum StaticTuplePlan {
    FusedScan,
    ReplayColumns,
}

impl StaticTuplePlan {
    pub(in crate::codegen) fn select(ir: &StructIR) -> Self {
        let replayable = ir
            .fields
            .iter()
            .filter_map(|field| match field {
                FieldPlan::Tuple(tuple) => replayable_tuple_terminal_count(tuple),
                FieldPlan::Column(_) => None,
            })
            .sum::<usize>();
        if replayable >= STATIC_TUPLE_REPLAY_MIN_TERMINALS {
            Self::ReplayColumns
        } else {
            Self::FusedScan
        }
    }

    pub(in crate::codegen) fn replays_root(self, field: &TupleField) -> bool {
        self == Self::ReplayColumns && tuple_group_has_replay_access(field.wrapper_shape())
    }

    pub(in crate::codegen) fn replays_terminal(
        self,
        leaf: PrimitiveLeaf<'_>,
        wrapper: &WrapperShape,
        has_replay_access: bool,
    ) -> bool {
        self == Self::ReplayColumns
            && has_replay_access
            && tuple_primitive_terminal_is_replayable(leaf, wrapper)
    }
}

pub(in crate::codegen) const fn tuple_group_has_replay_access(wrapper: &WrapperShape) -> bool {
    matches!(wrapper, WrapperShape::Leaf(shape) if shape.is_bare())
}

pub(in crate::codegen) const fn tuple_primitive_terminal_is_replayable(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
) -> bool {
    matches!(wrapper, WrapperShape::Leaf(_)) && leaf.evaluation_effect().allows_replay()
}

fn replayable_tuple_terminal_count(field: &TupleField) -> Option<usize> {
    if !tuple_group_has_replay_access(field.wrapper_shape()) {
        return None;
    }
    let count = replayable_terminal_count(field.elements());
    (count > 0).then_some(count)
}

fn replayable_terminal_count(elements: &crate::ir::NonEmpty<TupleNode>) -> usize {
    elements
        .iter()
        .map(|node| match node.kind() {
            TupleNodeKind::Leaf(common) => match common.leaf_spec().route() {
                TerminalLeafRoute::Primitive(leaf) => usize::from(
                    tuple_primitive_terminal_is_replayable(leaf, node.wrapper_shape()),
                ),
                TerminalLeafRoute::Nested(_) => 0,
            },
            TupleNodeKind::Tuple(children)
                if tuple_group_has_replay_access(node.wrapper_shape()) =>
            {
                replayable_terminal_count(children)
            }
            TupleNodeKind::Tuple(_) => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        AccessChain, AccessStep, DateTimeUnit, DurationSource, NumericKind, StringyBase,
        VecLayerSpec,
    };

    fn vec_shape(depth: usize, inner_access: AccessChain) -> VecLayers {
        VecLayers {
            layers: crate::ir::NonEmpty::from_vec(
                (0..depth)
                    .map(|_| VecLayerSpec {
                        access: AccessChain::empty(),
                    })
                    .collect(),
            )
            .expect("test list depth must be non-zero"),
            inner_access,
        }
    }

    fn parse_ir(input: &syn::DeriveInput) -> StructIR {
        crate::parser::parse_to_ir(input).expect("test input should lower to IR")
    }

    #[test]
    fn primitive_list_truth_table_is_typed() {
        let shallow = vec_shape(1, AccessChain::empty());
        let deep = vec_shape(2, AccessChain::empty());
        let optional_deep = vec_shape(2, AccessChain::empty().prepend_option());
        let mut smart_access = AccessChain::empty();
        smart_access.push(AccessStep::SmartPtr);
        let smart_deep = vec_shape(2, smart_access);

        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Numeric(NumericKind::I32),
                &deep,
                RowReplayCapability::Direct,
            ),
            PrimitiveListPolicy::ReplayRows(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Numeric(NumericKind::I32),
                &shallow,
                RowReplayCapability::Direct,
            ),
            PrimitiveListPolicy::CaptureSegments(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Numeric(NumericKind::I32),
                &deep,
                RowReplayCapability::Unavailable,
            ),
            PrimitiveListPolicy::CaptureSegments(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(PrimitiveLeaf::Bool, &deep, RowReplayCapability::Direct,),
            PrimitiveListPolicy::BulkSegments(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Bool,
                &deep,
                RowReplayCapability::Unavailable,
            ),
            PrimitiveListPolicy::BulkSegments(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Bool,
                &optional_deep,
                RowReplayCapability::Direct,
            ),
            PrimitiveListPolicy::ReplayRows(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Bool,
                &optional_deep,
                RowReplayCapability::Unavailable,
            ),
            PrimitiveListPolicy::CaptureSegments(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Bool,
                &smart_deep,
                RowReplayCapability::Direct,
            ),
            PrimitiveListPolicy::ReplayRows(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Bool,
                &smart_deep,
                RowReplayCapability::Unavailable,
            ),
            PrimitiveListPolicy::CaptureSegments(()),
        );
    }

    #[test]
    fn streaming_plan_owns_fallible_and_user_defined_evaluation() {
        let deep = vec_shape(2, AccessChain::empty());
        let custom_string = StringyBase::Struct(syn::parse_quote!(CustomString));
        for leaf in [
            PrimitiveLeaf::DateTime(DateTimeUnit::Nanoseconds),
            PrimitiveLeaf::NaiveDateTime(DateTimeUnit::Nanoseconds),
            PrimitiveLeaf::Duration {
                unit: DateTimeUnit::Microseconds,
                source: DurationSource::Chrono,
            },
            PrimitiveLeaf::Duration {
                unit: DateTimeUnit::Milliseconds,
                source: DurationSource::Std,
            },
            PrimitiveLeaf::Decimal {
                precision: 18,
                scale: 4,
            },
            PrimitiveLeaf::AsString,
            PrimitiveLeaf::AsStr(&custom_string),
        ] {
            assert_eq!(
                PrimitiveListPolicy::select(leaf, &deep, RowReplayCapability::Direct),
                PrimitiveListPolicy::StreamReserved(()),
                "{leaf:?}",
            );
        }
    }

    #[test]
    fn support_requirements_follow_the_selected_plan() {
        let bulk_bool = parse_ir(&syn::parse_quote! {
            struct BulkBool { values: Vec<Vec<bool>> }
        });
        let nullable_bool = parse_ir(&syn::parse_quote! {
            struct NullableBool { values: Vec<Option<bool>> }
        });
        let fixed = parse_ir(&syn::parse_quote! {
            struct Fixed { values: Vec<i32> }
        });
        let nullable_fixed = parse_ir(&syn::parse_quote! {
            struct NullableFixed { values: Vec<Option<i32>> }
        });

        let bulk = support_requirements(&bulk_bool);
        assert!(!bulk.push_reserved);
        assert!(!bulk.set_prepared_bitmap);
        let nullable = support_requirements(&nullable_bool);
        assert!(!nullable.push_reserved);
        assert!(nullable.set_prepared_bitmap);
        let fixed = support_requirements(&fixed);
        assert!(fixed.push_reserved);
        assert!(!fixed.set_prepared_bitmap);
        let nullable_fixed = support_requirements(&nullable_fixed);
        assert!(nullable_fixed.push_reserved);
        assert!(nullable_fixed.set_prepared_bitmap);
    }

    #[test]
    fn static_tuple_boundary_counts_only_eligible_terminals() {
        let fifteen = parse_ir(&syn::parse_quote! {
            struct Fifteen {
                values: (
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                ),
            }
        });
        let sixteen = parse_ir(&syn::parse_quote! {
            struct Sixteen {
                values: (
                    i64, i64, i64, i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64, i64, i64, i64,
                ),
            }
        });
        let seventeen = parse_ir(&syn::parse_quote! {
            struct Seventeen {
                values: (
                    i64, i64, i64, i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64, i64, i64, i64, i64,
                ),
            }
        });
        let mixed = parse_ir(&syn::parse_quote! {
            struct Mixed {
                values: (
                    i64, i64, i64, i64, i64, i64, i64, i64,
                    Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>,
                    Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>,
                ),
            }
        });
        let mixed_effects = parse_ir(&syn::parse_quote! {
            struct MixedEffects {
                values: (
                    Decimal,
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                    i64, i64, i64, i64, i64,
                ),
            }
        });
        let separate_groups = parse_ir(&syn::parse_quote! {
            struct SeparateGroups {
                a: (i64,), b: (i64,), c: (i64,), d: (i64,),
                e: (i64,), f: (i64,), g: (i64,), h: (i64,),
                i: (i64,), j: (i64,), k: (i64,), l: (i64,),
                m: (i64,), n: (i64,), o: (i64,), p: (i64,),
            }
        });
        let nested_bare = parse_ir(&syn::parse_quote! {
            struct NestedBare {
                values: (
                    (i64, i64, i64, i64, i64, i64, i64, i64),
                    (i64, i64, i64, i64, i64, i64, i64, i64),
                ),
            }
        });
        let nested_wrapped = parse_ir(&syn::parse_quote! {
            struct NestedWrapped {
                values: (
                    Option<(i64, i64, i64, i64, i64, i64, i64, i64)>,
                    (i64, i64, i64, i64, i64, i64, i64, i64),
                ),
            }
        });

        assert_eq!(
            StaticTuplePlan::select(&fifteen),
            StaticTuplePlan::FusedScan
        );
        assert_eq!(
            StaticTuplePlan::select(&sixteen),
            StaticTuplePlan::ReplayColumns,
        );
        assert_eq!(
            StaticTuplePlan::select(&seventeen),
            StaticTuplePlan::ReplayColumns,
        );
        assert_eq!(StaticTuplePlan::select(&mixed), StaticTuplePlan::FusedScan);
        assert_eq!(
            StaticTuplePlan::select(&mixed_effects),
            StaticTuplePlan::FusedScan,
        );
        assert_eq!(
            StaticTuplePlan::select(&separate_groups),
            StaticTuplePlan::ReplayColumns,
        );
        assert_eq!(
            StaticTuplePlan::select(&nested_bare),
            StaticTuplePlan::ReplayColumns,
        );
        assert_eq!(
            StaticTuplePlan::select(&nested_wrapped),
            StaticTuplePlan::FusedScan,
        );
        assert_eq!(STATIC_TUPLE_REPLAY_LANE_WIDTH, 8);
    }
}
