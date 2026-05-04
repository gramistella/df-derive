//! Centralized identifier generation for the encoder IR.
//!
//! Every identifier the encoder injects into generated code is built from
//! the `__df_derive_` prefix plus a structured suffix. Funneling every
//! identifier through this module gives a single place to look when adding
//! a new encoder or renaming an identifier that collides with a future user
//! identifier.
//!
//! The `__df_derive_some_` (flat-vec) versus `__df_derive_n_some_` (nested)
//! prefix divergence — and the matching `_total_layer_` versus
//! `_n_total_layer_` divergence — is intentional safety margin between the
//! two paths. Today the bind names cannot collide inside a single generated
//! function; the `n_` infix preserves the safety margin against future
//! emitter combinations that mix the paths.
//!
//! Visibility is `pub(super)` so naming details stay inside the encoder.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

// --- Per-field idents (indexed by field idx) ------------------------------

/// Owning `Vec<T>` / `Vec<Option<T>>` buffer for a primitive scalar field.
/// Holds `Vec<&str>` / `Vec<Option<&str>>` on the borrowing fast path.
pub(super) fn primitive_buf(idx: usize) -> Ident {
    format_ident!("__df_derive_buf_{}", idx)
}

/// `MutableBitmap` validity buffer for the
/// `is_direct_primitive_array_option_numeric_leaf` fast path. Paired with
/// `primitive_buf` (which holds `Vec<#native>` on that path) so the finisher
/// can build a `PrimitiveArray::new(dtype, vals, validity)` directly without
/// a `Vec<Option<T>>` second walk.
pub(super) fn primitive_validity(idx: usize) -> Ident {
    format_ident!("__df_derive_val_{}", idx)
}

/// Row counter for the `is_direct_view_option_string_leaf` fast path.
/// Indexes the pre-filled `MutableBitmap` so the per-row push only writes a
/// single byte for `None` rows via `set_unchecked`, instead of pushing both
/// `true` and `false` bits unconditionally.
pub(super) fn primitive_row_idx(idx: usize) -> Ident {
    format_ident!("__df_derive_ri_{}", idx)
}

/// Reused `String` scratch buffer for the `is_direct_view_to_string_leaf`
/// fast path. Paired with `primitive_buf` (which holds
/// `MutableBinaryViewArray<str>` on that path) so each row can clear-and-write
/// into the scratch via `Display::fmt` and then push the resulting `&str`
/// into the view array (which copies the bytes), avoiding a fresh per-row
/// `String` allocation.
pub(super) fn primitive_str_scratch(idx: usize) -> Ident {
    format_ident!("__df_derive_str_{}", idx)
}

/// Per-field local for the assembled Series produced by `vec_emit_decl` —
/// one per (field, depth) combination, namespaced by `idx` so two adjacent
/// fields don't collide.
pub(super) fn vec_field_series(idx: usize) -> Ident {
    format_ident!("__df_derive_field_series_{}", idx)
}

/// Synthesized per-row local for `wrap_multi_option_primitive`. Holds the
/// collapsed `Option<&T>` (or `Option<T>` after `.copied()`/`.cloned()`)
/// the inner option-leaf machinery consumes.
pub(super) fn multi_option_local(idx: usize) -> Ident {
    format_ident!("__df_derive_mo_{}", idx)
}

// --- Per-layer idents for the flat-vec push/scan path ---------------------

/// Per-layer offsets vec ident. Layer `i` is the `i`-th `Vec` from the
/// outside; layer `depth-1` is the innermost.
pub(super) fn vec_layer_offsets(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_off_{}", layer)
}

/// Per-layer outer-validity `MutableBitmap` ident.
pub(super) fn vec_layer_validity(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_val_{}", layer)
}

/// Per-layer iteration binding. Layer 0 binds the field access; deeper
/// layers bind the previous layer's iterator output.
pub(super) fn vec_layer_bind(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_bind_{}", layer)
}

/// Per-layer offsets buffer (frozen `OffsetsBuffer<i64>`).
pub(super) fn vec_layer_offsets_buf(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_off_buf_{}", layer)
}

/// Per-layer `LargeListArray` ident produced during the final-assemble step.
pub(super) fn vec_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_list_arr_{}", layer)
}

/// Per-layer total counter for the flat-vec precount loop. Layer `i` counts
/// child-lists produced by layer `i+1`.
pub(super) fn vec_layer_total(layer: usize) -> Ident {
    format_ident!("__df_derive_total_layer_{}", layer)
}

/// Token-form of the prefix used by the shared walker to construct
/// flat-vec outer-Some binds. The walker uses this prefix inside
/// `match collapsed { Some(<bind>) => ... }` arms (and the precount walker
/// inside `if let Some(<bind>) = ...` arms).
///
/// The `__df_derive_some_` prefix vs. the nested path's `__df_derive_n_some_`
/// is load-bearing — see the module docs.
pub(super) const VEC_OUTER_SOME_PREFIX: &str = "__df_derive_some_";

// --- Per-layer idents for the nested-struct push/scan path ----------------

/// Per-(field, layer) offsets vec ident for the nested encoder.
pub(super) fn nested_layer_offsets(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_off_{}_{}", idx, layer)
}

/// Per-(field, layer) frozen `OffsetsBuffer<i64>` ident for the nested encoder.
pub(super) fn nested_layer_offsets_buf(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_off_buf_{}_{}", idx, layer)
}

/// Per-(field, layer) `MutableBitmap` outer-validity ident for the nested
/// encoder.
pub(super) fn nested_layer_validity_mb(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_valmb_{}_{}", idx, layer)
}

/// Per-(field, layer) frozen `Bitmap` outer-validity ident for the nested
/// encoder.
pub(super) fn nested_layer_validity_bm(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_valbm_{}_{}", idx, layer)
}

/// Per-(field, layer) iteration binding for the nested encoder.
pub(super) fn nested_layer_bind(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_bind_{}_{}", idx, layer)
}

/// Per-layer total counter for the nested precount loop. Mirror of
/// [`vec_layer_total`] with the nested-path `n_` prefix.
pub(super) fn nested_layer_total(layer: usize) -> Ident {
    format_ident!("__df_derive_n_total_layer_{}", layer)
}

/// Token-form of the prefix used by the shared walker for the nested path.
/// The shared walker constructs per-layer binds inline as
/// `format_ident!("{prefix}{cur}")`. The `n_` infix is intentional safety
/// margin against the flat-vec path (see module docs).
pub(super) const NESTED_OUTER_SOME_PREFIX: &str = "__df_derive_n_some_";

/// Per-layer `LargeListArray` ident produced by the nested encoder's
/// `build_nested_layer_wrap`.
pub(super) fn nested_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_n_arr_{}", layer)
}

/// Innermost `ArrayRef` chunk pulled from the rechunked inner Series, used
/// as the seed of the layer-wrap stack.
pub(super) fn nested_inner_chunk() -> Ident {
    format_ident!("__df_derive_inner_chunk")
}

// --- Nested encoder per-field intermediates (one per field) ---------------

/// `Vec<&T>` flat ref accumulator for the nested encoder.
pub(super) fn nested_flat(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_flat_{}", idx)
}

/// `Vec<Option<IdxSize>>` per-element positions for the inner-Option
/// scatter case in the nested encoder.
pub(super) fn nested_positions(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_pos_{}", idx)
}

/// Inner `DataFrame` returned by `columnar_from_refs` in the nested encoder.
pub(super) fn nested_df(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_df_{}", idx)
}

/// `IdxCa` built from `positions` for the nested-encoder scatter case.
pub(super) fn nested_take(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_take_{}", idx)
}

/// Total inner-element count for the nested encoder (used by precount and
/// outer-list capacity).
pub(super) fn nested_total(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_total_{}", idx)
}

// --- Per-populator local idents (no parameter) ----------------------------

/// Outer iter binding for the populator's `for ... in items` ring.
pub(super) fn populator_iter() -> Ident {
    format_ident!("__df_derive_it")
}

/// Inner-leaf binding inside the deepest-layer for-loop. Bound to `&T`
/// (no inner-Option) or `Option<&T>` after the multi-Option collapse.
pub(super) fn leaf_value() -> Ident {
    format_ident!("__df_derive_v")
}

/// Raw inner-leaf binding before the multi-Option collapse (used only when
/// `inner_option_layers > 1`; the loop binding is `__df_derive_v_raw` and
/// then collapsed into [`leaf_value`]).
pub(super) fn leaf_value_raw() -> Ident {
    format_ident!("__df_derive_v_raw")
}

/// Innermost leaf array ident (the `PrimitiveArray<T>` / `Utf8ViewArray` /
/// `BooleanArray` value the layer stack wraps).
pub(super) fn leaf_arr() -> Ident {
    format_ident!("__df_derive_leaf_arr")
}

/// Total leaf-element counter for the flat-vec precount loop.
pub(super) fn total_leaves() -> Ident {
    format_ident!("__df_derive_total_leaves")
}

/// Bit-packed values bitmap ident for the bool inner-Option / bool-bare
/// vec encoders. Same name in both spec variants (the variants never
/// coexist in one block).
pub(super) fn bool_values() -> Ident {
    format_ident!("__df_derive_values")
}

/// Validity bitmap ident for the bool / numeric / string-like vec encoders.
/// Same name across spec variants (they never coexist in one block).
pub(super) fn bool_validity() -> Ident {
    format_ident!("__df_derive_validity")
}

/// Inner offsets vec ident for the depth-1 bool-bare fast path.
pub(super) fn bool_inner_offsets() -> Ident {
    format_ident!("__df_derive_inner_offsets")
}

/// Per-row inner-Option binding inside the nested-struct deepest leaf body.
pub(super) fn nested_maybe() -> Ident {
    format_ident!("__df_derive_maybe")
}

/// Closure parameter inside `collapse_options_to_ref`'s
/// `.and_then(|__df_derive_o| __df_derive_o.as_ref())` chain. Local to the
/// closure; centralized for consistency.
pub(super) fn collapse_option_param() -> Ident {
    format_ident!("__df_derive_o")
}

// --- Helpers for `quote!` ergonomics --------------------------------------

/// Returns a closure that produces the flat-vec precount counter token
/// stream for layer `i`. The precount loop allocates the counters; the
/// shared `shape_offsets_decls` / `shape_validity_decls` helpers consume
/// them via this closure.
pub(super) fn vec_layer_total_token(layer: usize) -> TokenStream {
    let id = vec_layer_total(layer);
    quote! { #id }
}

/// Mirror of [`vec_layer_total_token`] for the nested-encoder path.
pub(super) fn nested_layer_total_token(layer: usize) -> TokenStream {
    let id = nested_layer_total(layer);
    quote! { #id }
}
