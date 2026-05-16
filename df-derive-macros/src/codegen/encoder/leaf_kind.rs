//! Leaf payloads for the depth-N `Vec`-bearing emitter.
//!
//! Primitive leaves use per-element push into typed storage. Nested struct and
//! generic leaves collect references and materialize via `Columnar::columnar_from_refs`.

use proc_macro2::TokenStream;

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
