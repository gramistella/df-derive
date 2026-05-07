//! Nested-struct/generic encoder paths (`CollectThenBulk` leaves).
//!
//! Routes every nested-struct/generic wrapper shape — the bare `Nested`,
//! any `Option<...<Option<Nested>>>` stack, and every `Vec`-bearing stack
//! including deep nestings, mid-stack `Option`s, and outer-list validity —
//! through a single [`CollectThenBulk`] leaf and the unified emitter
//! [`super::emit::vec_emit_general`]. The depth-0 (`Leaf`) shape is the
//! degenerate case of the depth-N walker: no list-array wrap, the
//! all-absent arm uses `items.len()` instead of the precount `total`,
//! and the per-row scan body matches each row's optional access directly
//! rather than iterating an inner Vec.
//!
//! The invariant: every `LargeListArray::new` routes through the in-scope free
//! helper `__df_derive_assemble_list_series_unchecked` (defined at the top of
//! each derive's `const _: () = { ... };` scope), keeping `unsafe` out of any
//! `Self`-bearing impl method so `clippy::unsafe_derive_deserialize` stays
//! silent on downstream `#[derive(ToDataFrame, Deserialize)]` types.
//!
//! Every shape produces an `Encoder::Multi { columnar }` because the inner
//! `DataFrame` carries one column per inner schema entry of `T`. The block
//! pushes one Series per inner schema column onto the call site's `columns`
//! vec, with the parent name prefixed onto each inner column name.

use crate::ir::WrapperShape;
use proc_macro2::TokenStream;

use super::emit::vec_emit_general;
use super::leaf_kind::{CollectThenBulk, LeafKind};
use super::{BaseCtx, Encoder};

/// Per-call-site context for nested-struct/generic encoders. Carries the
/// type-as-path expression and the fully-qualified trait paths used in UFCS
/// calls (`<#ty as #columnar_trait>::columnar_from_refs`,
/// `<#ty as #to_df_trait>::schema`).
pub struct NestedLeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a TokenStream,
    pub to_df_trait: &'a TokenStream,
}

impl<'a> From<&NestedLeafCtx<'a>> for CollectThenBulk<'a> {
    fn from(ctx: &NestedLeafCtx<'a>) -> Self {
        Self {
            ty: ctx.ty,
            columnar_trait: ctx.columnar_trait,
            to_df_trait: ctx.to_df_trait,
            name: ctx.base.name,
            idx: ctx.base.idx,
        }
    }
}

/// Top-level dispatcher for the nested-struct/generic encoder paths. Every
/// wrapper shape the parser accepts — bare `Nested`, `Option<...<Nested>>`,
/// or any `Vec`-bearing stack — routes through the unified emitter via a
/// single [`CollectThenBulk`] leaf.
pub fn build_nested_encoder(wrapper: &WrapperShape, ctx: &NestedLeafCtx<'_>) -> Encoder {
    let ctb = CollectThenBulk::from(ctx);
    let kind = LeafKind::CollectThenBulk(ctb);
    let columnar = vec_emit_general(&kind, ctx.base.access, ctx.base.idx, wrapper);
    Encoder::Multi { columnar }
}
