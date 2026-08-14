//! Leaf payloads for the depth-N `Vec`-bearing emitter.
//!
//! Primitive leaves write typed storage at either element or source-segment
//! granularity. Nested struct and generic leaves collect references and
//! materialize via `Columnar::encode_ref_batch`.

use proc_macro2::TokenStream;

use crate::codegen::planner::PrimitiveListPlan;
pub(super) struct PrimitiveListCommon {
    pub(super) row_capacity: syn::Ident,
    pub(super) materialization: super::ctx::MaterializationTarget,
    pub(super) storage_decls: TokenStream,
    pub(super) leaf_arr_expr: TokenStream,
    pub(super) leaf_segment: syn::Ident,
    pub(super) extra_imports: TokenStream,
    pub(super) leaf_logical_dtype: TokenStream,
}

pub(super) struct StreamPrimitiveList {
    pub(super) common: PrimitiveListCommon,
    pub(super) fill_segment: TokenStream,
    pub(super) leaf_offsets_post_fill: TokenStream,
}

pub(super) struct CapturedPrimitiveList {
    pub(super) common: PrimitiveListCommon,
    pub(super) encode_support: syn::Path,
    pub(super) leaf_count: syn::Ident,
    pub(super) leaf_segments: syn::Ident,
    pub(super) fill_segment: TokenStream,
}

pub(super) struct GroupedPrimitiveList {
    pub(super) common: PrimitiveListCommon,
    pub(super) encode_support: syn::Path,
    pub(super) leaf_groups: syn::Ident,
    pub(super) fill_group: TokenStream,
}

pub(super) struct BulkPrimitiveList {
    pub(super) common: PrimitiveListCommon,
    pub(super) fill_segment: TokenStream,
    pub(super) leaf_offsets_post_fill: TokenStream,
}

pub(super) type PrimitiveListEncoding = PrimitiveListPlan<
    StreamPrimitiveList,
    CapturedPrimitiveList,
    GroupedPrimitiveList,
    BulkPrimitiveList,
>;

impl
    PrimitiveListPlan<
        StreamPrimitiveList,
        CapturedPrimitiveList,
        GroupedPrimitiveList,
        BulkPrimitiveList,
    >
{
    pub(super) const fn common(&self) -> &PrimitiveListCommon {
        match self {
            Self::StreamReserved(plan) => &plan.common,
            Self::CaptureSegments(plan) => &plan.common,
            Self::CaptureGroups(plan) => &plan.common,
            Self::BulkSegments(plan) => &plan.common,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct CollectThenBulk<'a> {
    pub row_capacity: &'a syn::Ident,
    pub ident_scope: super::idents::GeneratedIdentScope<'a>,
    pub sink: &'a syn::Ident,
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a syn::Path,
    pub columnar_spec_trait: &'a syn::Path,
    pub idx: usize,
}
