//! Encoding policy decisions shared by column and tuple lowering.
//!
//! This module chooses an execution shape; emitters remain responsible for
//! rendering that shape. Keeping the policy here prevents runtime replay,
//! storage selection, and hot-loop width from being inferred independently.

use crate::ir::{
    FieldPlan, PrimitiveLeaf, StructIR, TerminalLeafRoute, TupleField, TupleNode, TupleNodeKind,
    VecLayers, WrapperShape,
};

/// The sole primitive-list scheduling definition.
///
/// The unit specialization is the policy selected from IR. Lowering replaces
/// each unit with the distinct render payload required by that branch, so an
/// encoded plan cannot contain ingredients for a different branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum PrimitiveListPlan<Stream = (), Capture = (), Group = (), Bulk = ()> {
    /// Evaluate the leaf in the source pass and grow reserved storage.
    StreamReserved(Stream),
    /// Capture stable innermost list references and fill after the source pass.
    CaptureSegments(Capture),
    /// Capture stable penultimate list groups and fill their segments after
    /// the source pass.
    CaptureGroups(Group),
    /// Append real innermost Boolean slices through a whole-segment API.
    BulkSegments(Bulk),
}

pub(in crate::codegen) type PrimitiveListPolicy = PrimitiveListPlan<(), (), (), ()>;

impl PrimitiveListPlan<(), (), (), ()> {
    pub(in crate::codegen) fn for_wrapper(
        leaf: PrimitiveLeaf<'_>,
        wrapper: &WrapperShape,
    ) -> Option<Self> {
        match wrapper {
            WrapperShape::Vec(shape) => Some(Self::select(leaf, shape)),
            WrapperShape::Leaf(_) => None,
        }
    }

    pub(in crate::codegen) fn select(leaf: PrimitiveLeaf<'_>, shape: &VecLayers) -> Self {
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
        if shape.depth() >= 2 && shape.layers[shape.depth() - 1].access.is_empty() {
            return Self::CaptureGroups(());
        }
        Self::CaptureSegments(())
    }
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
        let mut optional_final_layer = vec_shape(2, AccessChain::empty());
        optional_final_layer.layers[1].access = AccessChain::empty().prepend_option();

        assert_eq!(
            PrimitiveListPolicy::select(PrimitiveLeaf::Numeric(NumericKind::I32), &deep),
            PrimitiveListPolicy::CaptureGroups(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(PrimitiveLeaf::Numeric(NumericKind::I32), &shallow),
            PrimitiveListPolicy::CaptureSegments(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(PrimitiveLeaf::Bool, &deep),
            PrimitiveListPolicy::BulkSegments(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(PrimitiveLeaf::Bool, &optional_deep),
            PrimitiveListPolicy::CaptureGroups(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(PrimitiveLeaf::Bool, &smart_deep),
            PrimitiveListPolicy::CaptureGroups(()),
        );
        assert_eq!(
            PrimitiveListPolicy::select(
                PrimitiveLeaf::Numeric(NumericKind::I32),
                &optional_final_layer,
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
                PrimitiveListPolicy::select(leaf, &deep),
                PrimitiveListPolicy::StreamReserved(()),
                "{leaf:?}",
            );
        }
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
