//! Nested-struct/generic encoder paths (`CollectThenBulk` leaves).
//!
//! Routes every nested-struct/generic wrapper shape — the bare `Nested`,
//! any `Option<...<Option<Nested>>>` stack, and every `Vec`-bearing stack
//! including deep nestings, mid-stack `Option`s, and outer-list validity —
//! through a single [`CollectThenBulk`] leaf and the unified emitter
//! [`super::emit::vec_emit_ctb`]. The depth-0 (`Leaf`) shape is the
//! degenerate case of the depth-N walker: no list-array wrap, the
//! all-absent arm uses `rows.len()` instead of the precount `total`,
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
//! checked batch carries one column per inner schema entry of `T`. The block
//! renames each validated inner column for its parent path and writes it
//! through the call site's `ColumnSink`.

use crate::ir::NestedNamePolicy;
use crate::ir::WrapperShape;
use proc_macro2::TokenStream;

use super::BaseCtx;
use super::emit::vec_emit_ctb;
use super::leaf_kind::CollectThenBulk;
use crate::codegen::external_paths::ExternalPaths;

/// Per-call-site context for nested-struct/generic encoders. Carries the
/// type-as-path expression and the fully-qualified trait paths used in UFCS
/// calls (`<#ty as #columnar_trait>::encode_batch`,
/// `<#ty as #columnar_spec_trait>::build_schema`).
pub struct NestedLeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub name_policy: &'a NestedNamePolicy,
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a syn::Path,
    pub columnar_spec_trait: &'a syn::Path,
    pub paths: &'a ExternalPaths,
}

impl<'a> From<&NestedLeafCtx<'a>> for CollectThenBulk<'a> {
    fn from(ctx: &NestedLeafCtx<'a>) -> Self {
        Self {
            rows: ctx.base.rows,
            sink: ctx.base.sink,
            ty: ctx.ty,
            columnar_trait: ctx.columnar_trait,
            columnar_spec_trait: ctx.columnar_spec_trait,
            name: ctx.base.name,
            name_policy: ctx.name_policy,
            idx: ctx.base.idx,
        }
    }
}

/// Top-level dispatcher for the nested-struct/generic encoder paths. Every
/// wrapper shape the parser accepts — bare `Nested`, `Option<...<Nested>>`,
/// or any `Vec`-bearing stack — routes through the unified emitter via a
/// single [`CollectThenBulk`] leaf.
pub fn build_nested_encoder(wrapper: &WrapperShape, ctx: &NestedLeafCtx<'_>) -> TokenStream {
    let ctb = CollectThenBulk::from(ctx);
    vec_emit_ctb(&ctb, ctx.base.access, ctx.base.idx, wrapper, ctx.paths)
}
