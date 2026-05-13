//! Leaf-kind abstraction for the depth-N `Vec`-bearing emitter.
//!
//! [`LeafKind`] dispatches the unified emitter
//! [`super::emit::vec_emit_general`] between two payload shapes:
//! `PerElementPush` (primitive leaves: typed-buffer push per row, one
//! `Series` out) and `CollectThenBulk` (nested struct / generic leaves:
//! gather `&T` refs, one `columnar_from_refs` call, per-inner-column
//! list-array stacking).
//!
//! See `docs/encoder-ir.md` for the conceptual model.

use proc_macro2::TokenStream;

use super::idents;
use super::shape_walk::OwnPolicy;

/// Per-element-push leaf payload — describes a primitive leaf's storage,
/// per-row push, leaf-array materialization, and post-push offsets-counter
/// expression. The unified emitter splices these into the shared depth-N
/// scaffolding.
pub(super) struct PerElementPush {
    /// Per-row push token stream. References `__df_derive_v` (bound by the
    /// for-loop) and pushes one element into the typed buffer.
    pub per_elem_push: TokenStream,
    /// Storage decls (buffers, validity bitmap, leaf-index counter as
    /// applicable). Spliced before the scan loop.
    pub storage_decls: TokenStream,
    /// Leaf array build expression — produces `let __df_derive_leaf_arr: ... = ...;`.
    pub leaf_arr_expr: TokenStream,
    /// Token expression producing the per-row offsets-push value (cast to
    /// i64). Differs by storage layout: `Vec`-len, MBVA-len, or a leaf-index
    /// counter for the bit-packed layouts.
    pub leaf_offsets_post_push: TokenStream,
    /// Optional `use Trait as _;` import to splice at the top of the emit
    /// scope. Used by the `Decimal` leaf so its `try_to_i128_mantissa`
    /// resolves on `&Decimal`.
    pub extra_imports: TokenStream,
    /// Per-leaf logical Polars dtype (e.g. `DataType::String`,
    /// `DataType::Decimal(p, s)`) for the outer-list assemble helper.
    pub leaf_logical_dtype: TokenStream,
}

/// Collect-then-bulk leaf payload — describes a nested-struct/generic leaf's
/// storage (flat ref vec, optional positions vec), the dispatch arm count
/// (2 or 4), and the trait/type plumbing the post-scan materialization
/// needs to call `<T as Columnar>::columnar_from_refs(...)`.
pub(super) struct CollectThenBulk<'a> {
    /// `<#ty>::columnar_from_refs(&flat)` target type.
    pub ty: &'a TokenStream,
    /// `<#ty as #columnar_trait>::columnar_from_refs` trait path.
    pub columnar_trait: &'a TokenStream,
    /// `<#ty as #to_df_trait>::schema()` trait path.
    pub to_df_trait: &'a TokenStream,
    /// Parent-field name; prefixed onto each inner schema column name.
    pub name: &'a str,
    /// Field index — namespaces the per-field idents
    /// (`__df_derive_gen_flat_<idx>`, etc).
    pub idx: usize,
}

/// Per-leaf-kind dispatch for the depth-N `Vec`-bearing emitter.
///
/// `PerElementPush` covers numeric / string / bool / decimal / datetime
/// primitive leaves: one `Series` push at the end, freeze interleaved with
/// each layer's wrap, single dispatch arm.
///
/// `CollectThenBulk` covers nested user structs and generic type parameters:
/// gathers `&T` refs (and optional `Option<IdxSize>` positions), dispatches
/// on `(total, flat.len())` to 2 or 4 branches, and per branch iterates the
/// inner schema and emits per-column list-array stacking.
pub(super) enum LeafKind<'a> {
    PerElementPush(PerElementPush),
    CollectThenBulk(CollectThenBulk<'a>),
}

impl LeafKind<'_> {
    /// Whether this leaf kind hoists offsets/validity freezes above branch
    /// dispatch (collect-then-bulk) or interleaves them per-layer with each
    /// `LargeListArray::new` (per-element-push).
    pub(super) const fn freeze_hoisted(&self) -> bool {
        matches!(self, Self::CollectThenBulk(_))
    }

    /// Outer-some-prefix used by the scan walker for this leaf kind.
    /// `__df_derive_some_` for per-element-push; `__df_derive_n_some_` for
    /// collect-then-bulk.
    pub(super) const fn scan_outer_some_prefix(&self) -> &'static str {
        match self {
            Self::PerElementPush(_) => idents::VEC_OUTER_SOME_PREFIX,
            Self::CollectThenBulk(_) => idents::NESTED_OUTER_SOME_PREFIX,
        }
    }

    /// Outer-some-prefix used by the precount walker for this leaf kind.
    /// Per-element-push uses the same outer-Some prefix
    /// (`__df_derive_some_`) for both precount and scan because for-loop
    /// pattern bindings are scoped to the loop body, so the two for-loops
    /// never collide on shared idents. Collect-then-bulk uses two distinct
    /// prefixes (`__df_derive_n_pre_some_` for precount,
    /// `__df_derive_n_some_` for scan).
    pub(super) const fn precount_outer_some_prefix(&self) -> &'static str {
        match self {
            Self::PerElementPush(_) => idents::VEC_OUTER_SOME_PREFIX,
            Self::CollectThenBulk(_) => idents::NESTED_PRE_OUTER_SOME_PREFIX,
        }
    }

    /// Per-layer offsets-buffer ownership policy. The per-element-push path
    /// uses each frozen buffer in exactly one `LargeListArray::new` site
    /// (`OwnPolicy::Move`); the collect-then-bulk path's four-arm dispatch
    /// reuses the same buffer per arm and per inner-schema-column, so it
    /// must clone (`OwnPolicy::Clone`).
    pub(super) const fn layer_own_policy<'b>(&self, buf_id: &'b syn::Ident) -> OwnPolicy<'b> {
        match self {
            Self::PerElementPush(_) => OwnPolicy::Move(buf_id),
            Self::CollectThenBulk(_) => OwnPolicy::Clone(buf_id),
        }
    }
}
