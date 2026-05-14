//! Centralized identifier generation for the encoder IR.
//!
//! Every identifier the macro injects into generated code is built from
//! the `__df_derive_` prefix plus a structured suffix. Funneling every
//! identifier through this module gives a single place to look when adding
//! a new emitter or renaming an identifier that collides with a future user
//! identifier.
//!
//! The `__df_derive_some_` (flat-vec) versus `__df_derive_n_some_` (nested)
//! prefix divergence — and the matching `_total_layer_` versus
//! `_n_total_layer_` divergence — is intentional safety margin between the
//! two paths. Today the bind names cannot collide inside a single generated
//! function; the `n_` infix preserves the safety margin against future
//! emitter combinations that mix the paths.
//!
//! Visibility is `pub(in crate::codegen)` so the encoder's siblings under
//! `src/codegen/` (the columnar/strategy/nested-schema entry points and the
//! per-derive helper definition in `mod.rs`) can route their literals
//! through the same registry. The contract is repo-wide: every
//! `__df_derive_*` identifier the macro emits comes from this file.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

// --- Per-field idents (indexed by field idx) ------------------------------

/// Owning `Vec<T>` / `Vec<Option<T>>` buffer for a primitive scalar field.
/// Holds `Vec<&str>` / `Vec<Option<&str>>` on the borrowing fast path.
pub(in crate::codegen) fn primitive_buf(idx: usize) -> Ident {
    format_ident!("__df_derive_buf_{}", idx)
}

/// `MutableBitmap` validity buffer for the option-bearing primitive leaves
/// (numeric / `Decimal` / `DateTime` / bool / string-like). Paired with
/// `primitive_buf` (which holds `Vec<#native>` for numeric leaves and an
/// `MBVA<str>` for string-like leaves) so the finisher can build a
/// `PrimitiveArray::new(dtype, vals, validity)` directly without a
/// `Vec<Option<T>>` second walk.
pub(in crate::codegen) fn primitive_validity(idx: usize) -> Ident {
    format_ident!("__df_derive_val_{}", idx)
}

/// Row counter paired with a pre-filled `MutableBitmap` validity buffer for
/// the option-bearing string-like / bool primitive leaves. Indexes the
/// pre-filled bitmap so the per-row push only flips a single bit on the
/// `None` rows, instead of pushing both `true` and `false` bits
/// unconditionally.
pub(in crate::codegen) fn primitive_row_idx(idx: usize) -> Ident {
    format_ident!("__df_derive_ri_{}", idx)
}

/// Reused `String` scratch buffer for the `to_string` (`as_string`) leaf.
/// Paired with `primitive_buf` (which holds `MutableBinaryViewArray<str>`
/// for that leaf) so each row can clear-and-write into the scratch via
/// `Display::fmt` and then push the resulting `&str` into the view array
/// (which copies the bytes), avoiding a fresh per-row `String` allocation.
pub(in crate::codegen) fn primitive_str_scratch(idx: usize) -> Ident {
    format_ident!("__df_derive_str_{}", idx)
}

/// Per-field local for the assembled Series produced by `emit::pep_emit` —
/// one per (field, depth) combination, namespaced by `idx` so two adjacent
/// fields don't collide.
pub(in crate::codegen) fn vec_field_series(idx: usize) -> Ident {
    format_ident!("__df_derive_field_series_{}", idx)
}

/// Synthesized per-row local for `wrap_multi_option_primitive`. Holds the
/// collapsed `Option<&T>` (or `Option<T>` after `.copied()`/`.cloned()`)
/// the inner option-leaf machinery consumes.
pub(in crate::codegen) fn multi_option_local(idx: usize) -> Ident {
    format_ident!("__df_derive_mo_{}", idx)
}

// --- Per-layer idents for the flat-vec push/scan path ---------------------

/// Per-layer offsets vec ident. Layer `i` is the `i`-th `Vec` from the
/// outside; layer `depth-1` is the innermost.
pub(in crate::codegen) fn vec_layer_offsets(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_off_{}", layer)
}

/// Per-layer outer-validity `MutableBitmap` ident.
pub(in crate::codegen) fn vec_layer_validity(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_val_{}", layer)
}

/// Per-layer iteration binding. Layer 0 binds the field access; deeper
/// layers bind the previous layer's iterator output.
pub(in crate::codegen) fn vec_layer_bind(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_bind_{}", layer)
}

/// Per-layer offsets buffer (frozen `OffsetsBuffer<i64>`).
pub(in crate::codegen) fn vec_layer_offsets_buf(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_off_buf_{}", layer)
}

/// Per-layer frozen `Bitmap` outer-validity ident for the flat-vec encoder.
/// Mirror of [`nested_layer_validity_bm`] for the flat-vec path; the freeze
/// is hoisted out of the wrap loop so the assemble helper can clone the
/// pre-frozen buffer rather than consume the `MutableBitmap` directly.
pub(in crate::codegen) fn vec_layer_validity_bm(layer: usize) -> Ident {
    format_ident!("__df_derive_layer_val_bm_{}", layer)
}

/// Per-layer `LargeListArray` ident produced during the final-assemble step.
pub(in crate::codegen) fn vec_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_list_arr_{}", layer)
}

/// Per-layer total counter for the flat-vec precount loop. Layer `i` counts
/// child-lists produced by layer `i+1`.
pub(in crate::codegen) fn vec_layer_total(layer: usize) -> Ident {
    format_ident!("__df_derive_total_layer_{}", layer)
}

/// Token-form of the prefix used by the shared walker to construct
/// flat-vec outer-Some binds. The walker uses this prefix inside
/// `match collapsed { Some(<bind>) => ... }` arms (and the precount walker
/// inside `if let Some(<bind>) = ...` arms).
///
/// The `__df_derive_some_` prefix vs. the nested path's `__df_derive_n_some_`
/// is load-bearing — see the module docs.
pub(in crate::codegen) const VEC_OUTER_SOME_PREFIX: &str = "__df_derive_some_";

// --- Per-layer idents for the nested-struct push/scan path ----------------

/// Per-(field, layer) offsets vec ident for the nested encoder.
pub(in crate::codegen) fn nested_layer_offsets(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_off_{}_{}", idx, layer)
}

/// Per-(field, layer) frozen `OffsetsBuffer<i64>` ident for the nested encoder.
pub(in crate::codegen) fn nested_layer_offsets_buf(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_off_buf_{}_{}", idx, layer)
}

/// Per-(field, layer) `MutableBitmap` outer-validity ident for the nested
/// encoder.
pub(in crate::codegen) fn nested_layer_validity_mb(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_valmb_{}_{}", idx, layer)
}

/// Per-(field, layer) frozen `Bitmap` outer-validity ident for the nested
/// encoder.
pub(in crate::codegen) fn nested_layer_validity_bm(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_valbm_{}_{}", idx, layer)
}

/// Per-(field, layer) iteration binding for the nested encoder.
pub(in crate::codegen) fn nested_layer_bind(idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_n_bind_{}_{}", idx, layer)
}

/// Per-layer total counter for the nested precount loop. Mirror of
/// [`vec_layer_total`] with the nested-path `n_` prefix.
pub(in crate::codegen) fn nested_layer_total(layer: usize) -> Ident {
    format_ident!("__df_derive_n_total_layer_{}", layer)
}

/// Token-form of the prefix used by the shared walker for the nested path.
/// The shared walker constructs per-layer binds inline as
/// `format_ident!("{prefix}{cur}")`. The `n_` infix is intentional safety
/// margin against the flat-vec path (see module docs).
pub(in crate::codegen) const NESTED_OUTER_SOME_PREFIX: &str = "__df_derive_n_some_";

/// Token-form of the prefix used by the nested-path precount walker. Distinct
/// from [`NESTED_OUTER_SOME_PREFIX`] (the scan walker's prefix) so the two
/// walkers' loops can coexist inside the same generated block without their
/// outer-Some binds shadowing one another.
pub(in crate::codegen) const NESTED_PRE_OUTER_SOME_PREFIX: &str = "__df_derive_n_pre_some_";

/// Per-layer `LargeListArray` ident produced by the nested encoder's
/// `emit::ctb_layer_wrap`.
pub(in crate::codegen) fn nested_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_n_arr_{}", layer)
}

/// Innermost `ArrayRef` chunk pulled from the rechunked inner Series, used
/// as the seed of the layer-wrap stack.
pub(in crate::codegen) fn nested_inner_chunk() -> Ident {
    format_ident!("__df_derive_inner_chunk")
}

/// Seed-array arrow dtype local captured before the typed leaf array is
/// boxed into an `ArrayRef`.
pub(in crate::codegen) fn seed_arrow_dtype() -> Ident {
    format_ident!("__df_derive_seed_dt")
}

// --- Nested encoder per-field intermediates (one per field) ---------------

/// `Vec<&T>` flat ref accumulator for the nested encoder.
pub(in crate::codegen) fn nested_flat(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_flat_{}", idx)
}

/// `Vec<Option<IdxSize>>` per-element positions for the inner-Option
/// scatter case in the nested encoder.
pub(in crate::codegen) fn nested_positions(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_pos_{}", idx)
}

/// Inner `DataFrame` returned by `columnar_from_refs` in the nested encoder.
pub(in crate::codegen) fn nested_df(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_df_{}", idx)
}

/// `IdxCa` built from `positions` for the nested-encoder scatter case.
pub(in crate::codegen) fn nested_take(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_take_{}", idx)
}

/// Total inner-element count for the nested encoder (used by precount and
/// outer-list capacity).
pub(in crate::codegen) fn nested_total(idx: usize) -> Ident {
    format_ident!("__df_derive_gen_total_{}", idx)
}

// --- Per-populator local idents (no parameter) ----------------------------

/// Outer iter binding for the populator's `for ... in items` ring.
pub(in crate::codegen) fn populator_iter() -> Ident {
    format_ident!("__df_derive_it")
}

/// Inner-leaf binding inside the deepest-layer for-loop. Bound to `&T`
/// (no inner-Option) or `Option<&T>` after the multi-Option collapse.
pub(in crate::codegen) fn leaf_value() -> Ident {
    format_ident!("__df_derive_v")
}

/// Raw inner-leaf binding before the multi-Option collapse (used only when
/// `inner_option_layers > 1`; the loop binding is `__df_derive_v_raw` and
/// then collapsed into [`leaf_value`]).
pub(in crate::codegen) fn leaf_value_raw() -> Ident {
    format_ident!("__df_derive_v_raw")
}

/// Innermost leaf array ident (the `PrimitiveArray<T>` / `Utf8ViewArray` /
/// `BooleanArray` value the layer stack wraps).
pub(in crate::codegen) fn leaf_arr() -> Ident {
    format_ident!("__df_derive_leaf_arr")
}

/// Total leaf-element counter for the flat-vec precount loop.
pub(in crate::codegen) fn total_leaves() -> Ident {
    format_ident!("__df_derive_total_leaves")
}

/// Bit-packed values bitmap ident for the bool inner-Option / bool-bare
/// vec encoders. Same name in both spec variants (the variants never
/// coexist in one block).
pub(in crate::codegen) fn bool_values() -> Ident {
    format_ident!("__df_derive_values")
}

/// Validity bitmap ident for the bool / numeric / string-like vec encoders.
/// Same name across spec variants (they never coexist in one block).
pub(in crate::codegen) fn bool_validity() -> Ident {
    format_ident!("__df_derive_validity")
}

/// Inner offsets vec ident for the depth-1 bool-bare fast path.
pub(in crate::codegen) fn bool_inner_offsets() -> Ident {
    format_ident!("__df_derive_inner_offsets")
}

/// Checked `i64` list offset local used before pushing into offsets vecs.
pub(in crate::codegen) fn list_offset() -> Ident {
    format_ident!("__df_derive_offset")
}

/// Flat values buffer ident for the bulk-vec leaf paths: `Vec<#native>` for
/// numeric / decimal / datetime spec variants, and `Vec<bool>` for the
/// depth-1 bool-bare fast path. Same name across variants — they never
/// coexist in one block.
pub(in crate::codegen) fn vec_flat() -> Ident {
    format_ident!("__df_derive_flat")
}

/// `MutableBinaryViewArray<str>` accumulator ident for the bulk-vec
/// string-like leaf paths (`String`, `to_string`, `as_str`).
pub(in crate::codegen) fn vec_view_buf() -> Ident {
    format_ident!("__df_derive_view_buf")
}

/// Per-leaf index counter ident for bulk-vec leaves that pre-fill a values /
/// validity bitmap and need a running index for `set(idx, ...)` calls
/// (bool inner-Option, bool-bare, string-like with inner-Option).
pub(in crate::codegen) fn vec_leaf_idx() -> Ident {
    format_ident!("__df_derive_leaf_idx")
}

/// Local `MutableBitmap` builder ident used inside the pre-filled-bitmap
/// init blocks (`let mut <bitmap> = { let mut __df_derive_b = ...; b }`).
/// Local to each builder block; centralized for consistency.
pub(in crate::codegen) fn bitmap_builder() -> Ident {
    format_ident!("__df_derive_b")
}

/// Frozen `OffsetsBuffer<i64>` local for the depth-1 bool-bare fast path.
/// Sole call site, kept here so the safety-net scanner finds the literal in
/// exactly one place.
pub(in crate::codegen) fn bool_bare_offsets_buf() -> Ident {
    format_ident!("__df_derive_offsets_buf")
}

/// `LargeListArray` local for the depth-1 bool-bare fast path. Like
/// [`bool_bare_offsets_buf`], a single-call-site ident centralized so the
/// scanner sees one definition, not a literal.
pub(in crate::codegen) fn bool_bare_list_arr() -> Ident {
    format_ident!("__df_derive_list_arr")
}

/// Per-field `with_name(...)`-renamed Series local pushed onto `columns` in
/// the bulk-vec and nested-struct columnar blocks. Same name across both
/// paths (the two paths' blocks are independent scopes).
pub(in crate::codegen) fn field_named_series() -> Ident {
    format_ident!("__df_derive_named")
}

/// Per-row `format!`-built prefixed column name ident inside the
/// nested-struct `for col in schema` loop body.
pub(in crate::codegen) fn nested_prefixed_name() -> Ident {
    format_ident!("__df_derive_prefixed")
}

/// Inner schema-column-name binding inside the nested-struct
/// `for (name, dtype) in schema()?` loop. The schema name is exposed as
/// `&str` and used to look up the inner Series via `df.column(name)`.
pub(in crate::codegen) fn nested_col_name() -> Ident {
    format_ident!("__df_derive_col_name")
}

/// Inner schema-column-dtype binding inside the nested-struct
/// `for (name, dtype) in schema()?` loop. Borrowed as `&DataType` so the
/// per-column expressions can pass it to typed-null builders.
pub(in crate::codegen) fn nested_col_dtype() -> Ident {
    format_ident!("__df_derive_dtype")
}

/// Per-column inner Series local inside the nested-struct columnar block —
/// pulled from the inner `DataFrame`, then renamed and pushed.
pub(in crate::codegen) fn nested_inner_series() -> Ident {
    format_ident!("__df_derive_inner")
}

/// Per-column "full" inner Series local for the scatter-via-take branch of
/// the nested-struct dispatch. Bound to `df.column(...)?.as_materialized_series()`
/// then `take(&idx)?`-ed into the final Series.
pub(in crate::codegen) fn nested_inner_full() -> Ident {
    format_ident!("__df_derive_inner_full")
}

/// Per-column rechunked inner Series local in the depth-N nested-Vec
/// encoder's `emit::ctb_layer_wrap` — feeds the layer-wrap stack a
/// single contiguous `ArrayRef`.
pub(in crate::codegen) fn nested_inner_col() -> Ident {
    format_ident!("__df_derive_inner_col")
}

/// Per-column post-rechunk Series local in `emit::ctb_layer_wrap`. Holds
/// the result of `inner_col.rechunk()` so the layer-wrap stack can pull
/// `chunks()[0].clone()` as the seed `ArrayRef`.
pub(in crate::codegen) fn nested_inner_rech() -> Ident {
    format_ident!("__df_derive_inner_rech")
}

/// Mutable `DataType` accumulator inside the schema/empty-frame helpers'
/// per-layer `List<>` wrap loop. Each iteration replaces it with
/// `DataType::List(Box::new(prev))`, building up the runtime dtype that
/// matches the field's `Vec<…<Vec<T>>>` nesting.
pub(in crate::codegen) fn schema_wrapped_dtype() -> Ident {
    format_ident!("__df_derive_wrapped")
}

/// Free-function helper emitted at the top of each derive's per-derive
/// `const _: () = { ... };` scope. The helper holds the only
/// `Series::from_chunks_and_dtype_unchecked` `unsafe` block in the
/// generated code; every bulk-list emit site routes through it.
pub(in crate::codegen) fn assemble_helper() -> Ident {
    format_ident!("__df_derive_assemble_list_series_unchecked")
}

/// Per-derive internal wrapper emitted beside [`assemble_helper`]. It owns
/// the final `LargeListArray` plus its logical Polars `DataType` and is the
/// generated-code abstraction that checks their physical compatibility before
/// the unchecked Series construction.
pub(in crate::codegen) fn list_assembly() -> Ident {
    format_ident!("__DfDeriveListAssembly")
}

/// Per-derive helper that validates a manual nested `Columnar` implementation
/// returned exactly one row per gathered nested value before generated code
/// reads columns from it.
pub(in crate::codegen) fn validate_nested_frame() -> Ident {
    format_ident!("__df_derive_validate_nested_frame")
}

/// Per-derive helper that validates a nested `DataFrame` column's actual dtype
/// matches the dtype declared by the nested type's schema before the dtype is
/// reused for list assembly or positional `take`.
pub(in crate::codegen) fn validate_nested_column_dtype() -> Ident {
    format_ident!("__df_derive_validate_nested_column_dtype")
}

/// Per-derive helper used to eagerly assert that concrete
/// `#[df_derive(as_str)]` custom paths implement `AsRef<str>`.
pub(in crate::codegen) fn as_ref_str_assert_helper() -> Ident {
    format_ident!("__df_derive_assert_as_ref_str")
}

/// Per-derive helper used to eagerly assert that concrete
/// `#[df_derive(as_string)]` custom paths implement `Display`.
pub(in crate::codegen) fn display_assert_helper() -> Ident {
    format_ident!("__df_derive_assert_display")
}

/// Per-row inner-Option binding inside the nested-struct deepest leaf body.
pub(in crate::codegen) fn nested_maybe() -> Ident {
    format_ident!("__df_derive_maybe")
}

/// Closure parameter inside `collapse_options_to_ref`'s
/// `.and_then(|__df_derive_o| __df_derive_o.as_ref())` chain. Local to the
/// closure; centralized for consistency.
pub(in crate::codegen) fn collapse_option_param() -> Ident {
    format_ident!("__df_derive_o")
}

// --- Tuple-emitter idents -------------------------------------------------
//
// The tuple emitter operates on a per-field, per-element-column basis with
// tuple-specific adapters over the shared shape walker (see `super::tuple`).
// It uses a distinct `__df_derive_t_*` prefix family to keep its locals from
// colliding with the primitive/nested encoder paths inside the same generated
// function. Every ident the tuple emitter introduces must come from this
// section.

/// Per-(field, layer) offsets vec ident for the tuple emitter.
pub(in crate::codegen) fn tuple_layer_offsets(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_off_{}_{}", field_idx, layer)
}

/// Per-(field, layer) frozen `OffsetsBuffer<i64>` ident for the tuple emitter.
pub(in crate::codegen) fn tuple_layer_offsets_buf(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_off_buf_{}_{}", field_idx, layer)
}

/// Per-(field, layer) `MutableBitmap` outer-validity ident for the tuple
/// emitter.
pub(in crate::codegen) fn tuple_layer_validity_mb(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_valmb_{}_{}", field_idx, layer)
}

/// Per-(field, layer) frozen `Bitmap` outer-validity ident for the tuple
/// emitter.
pub(in crate::codegen) fn tuple_layer_validity_bm(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_valbm_{}_{}", field_idx, layer)
}

/// Per-(field, layer) iteration binding for the tuple emitter.
pub(in crate::codegen) fn tuple_layer_bind(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_bind_{}_{}", field_idx, layer)
}

/// Per-(field, layer) precount counter ident for the tuple emitter.
pub(in crate::codegen) fn tuple_layer_total(field_idx: usize, layer: usize) -> Ident {
    format_ident!("__df_derive_t_total_{}_{}", field_idx, layer)
}

/// Per-(field, layer) `LargeListArray` ident produced by the tuple
/// emitter's materialization step.
pub(in crate::codegen) fn tuple_layer_list_arr(layer: usize) -> Ident {
    format_ident!("__df_derive_t_arr_{}", layer)
}

/// Token-form of the prefix used by the shared walker for tuple-field
/// projection scans.
pub(in crate::codegen) const TUPLE_OUTER_SOME_PREFIX: &str = "__df_derive_t_some_";

/// Token-form of the prefix used by the shared walker for tuple-field
/// projection precounts. Distinct from [`TUPLE_OUTER_SOME_PREFIX`] so the
/// scan and precount loops can coexist without name shadowing.
pub(in crate::codegen) const TUPLE_PRE_OUTER_SOME_PREFIX: &str = "__df_derive_t_pre_some_";

/// Closure parameter for the tuple-element projection lambda
/// (`.map(|__df_derive_t| ...)`). Local to the lambda; centralized so the
/// safety-net scanner sees a single source.
pub(in crate::codegen) fn tuple_proj_param() -> Ident {
    format_ident!("__df_derive_t")
}

/// Per-row inner-Option binding for the tuple emitter's nested-element
/// flat-collect path. Equivalent to [`nested_maybe`] but namespaced so the
/// tuple emitter's collect-then-bulk pieces don't collide with the
/// nested-struct encoder's locals when both fire in the same generated
/// function.
pub(in crate::codegen) fn tuple_nested_inner_v() -> Ident {
    format_ident!("__df_derive_inner_v")
}

// --- Helpers for `quote!` ergonomics --------------------------------------

/// Returns a closure that produces the flat-vec precount counter token
/// stream for layer `i`. The precount loop allocates the counters; the
/// shared `shape_offsets_decls` / `shape_validity_decls` helpers consume
/// them via this closure.
pub(in crate::codegen) fn vec_layer_total_token(layer: usize) -> TokenStream {
    let id = vec_layer_total(layer);
    quote! { #id }
}

/// Mirror of [`vec_layer_total_token`] for the nested-encoder path.
pub(in crate::codegen) fn nested_layer_total_token(layer: usize) -> TokenStream {
    let id = nested_layer_total(layer);
    quote! { #id }
}
