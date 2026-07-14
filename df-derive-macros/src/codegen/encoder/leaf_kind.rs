//! Leaf payloads for the depth-N `Vec`-bearing emitter.
//!
//! Primitive leaves use per-element push into typed storage. Nested struct and
//! generic leaves collect references and materialize via `Columnar::encode_batch`.

use proc_macro2::TokenStream;

use crate::ir::NestedNamePolicy;

#[derive(Clone)]
pub(super) struct PerElementPush {
    pub row_capacity: syn::Ident,
    pub schedule: PrimitiveListSchedule,
    /// Writes one leaf after its schedule has established the storage slot.
    pub write_leaf: TokenStream,
    pub storage_decls: TokenStream,
    pub leaf_arr_expr: TokenStream,
    pub extra_imports: TokenStream,
    pub leaf_logical_dtype: TokenStream,
}

#[derive(Clone)]
pub(super) enum PrimitiveListSchedule {
    /// Fill while the source row is current so fallible or user-defined leaf
    /// evaluation preserves iterator-consumption and evaluation order.
    Immediate {
        prepare_segment: TokenStream,
        leaf_offsets_post_push: TokenStream,
    },
    /// Fill one complete innermost list segment through a bulk leaf API.
    ImmediateSegments {
        leaf_segment: syn::Ident,
        write_segment: TokenStream,
        leaf_offsets_post_push: TokenStream,
    },
    /// Replay the shared source-row references after exact list cardinality is
    /// known. This bounds staging by the input row count for deep lists.
    DeferredRows {
        shape_counts: syn::Ident,
        leaf_offsets_post_push: TokenStream,
        row: syn::Ident,
        replay_rows: syn::Ident,
    },
    /// Record stable leaf-Vec references when no enclosing row replay is
    /// available, then fill exact-sized storage after the source pass.
    DeferredSegments {
        leaf_count: syn::Ident,
        leaf_segments: syn::Ident,
        leaf_segment: syn::Ident,
    },
}

#[derive(Clone, Copy)]
pub(super) struct CollectThenBulk<'a> {
    pub row_capacity: &'a syn::Ident,
    pub sink: &'a syn::Ident,
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a syn::Path,
    pub columnar_spec_trait: &'a syn::Path,
    pub name: &'a str,
    pub name_policy: &'a NestedNamePolicy,
    pub idx: usize,
}
