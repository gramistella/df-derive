//! Leaf-kind abstraction for the depth-N `Vec`-bearing emitter.
//!
//! [`LeafKind`] lets the shape-aware emitters dispatch between two payload shapes:
//! `PerElementPush` (primitive leaves: typed-buffer push per row, one
//! `Series` out) and `CollectThenBulk` (nested struct / generic leaves:
//! gather `&T` refs, one `columnar_from_refs` call, per-inner-column
//! list-array stacking).

use proc_macro2::TokenStream;

use super::idents;

#[derive(Clone)]
pub(super) struct PerElementPush {
    pub per_elem_push: TokenStream,
    pub storage_decls: TokenStream,
    pub leaf_arr_expr: TokenStream,
    pub leaf_offsets_post_push: TokenStream,
    pub extra_imports: TokenStream,
    pub leaf_logical_dtype: TokenStream,
}

#[derive(Clone, Copy)]
pub(super) struct CollectThenBulk<'a> {
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a syn::Path,
    pub to_df_trait: &'a syn::Path,
    pub name: &'a str,
    pub idx: usize,
}

pub(super) enum LeafKind {
    PerElementPush,
    CollectThenBulk,
}

impl LeafKind {
    pub(super) const fn scan_outer_some_prefix(&self) -> &'static str {
        match self {
            Self::PerElementPush => idents::VEC_OUTER_SOME_PREFIX,
            Self::CollectThenBulk => idents::NESTED_OUTER_SOME_PREFIX,
        }
    }

    pub(super) const fn precount_outer_some_prefix(&self) -> &'static str {
        match self {
            Self::PerElementPush => idents::VEC_OUTER_SOME_PREFIX,
            Self::CollectThenBulk => idents::NESTED_PRE_OUTER_SOME_PREFIX,
        }
    }
}
