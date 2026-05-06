//! Leaf-kind abstraction for the depth-N `Vec`-bearing emitter.
//!
//! The encoder IR has two leaf kinds (per `docs/encoder-ir.md`):
//!
//! - **Per-element-push**: primitive leaves (numeric, `String`, `Bool`,
//!   `Decimal`, `DateTime`, ...) accumulate one value per row inside a
//!   tight loop and produce one Polars `Series`.
//! - **Collect-then-bulk**: nested user structs and generic `T` parameters
//!   accumulate `&Inner` references across all rows, then dispatch a single
//!   `<Inner as Columnar>::columnar_from_refs(&refs)` call to materialize
//!   every inner column at once.
//!
//! Both kinds share the depth-N walker primitives in [`super::shape_walk`]
//! (precount, scan, layer-idents, offsets-decl, validity-decl, list-array
//! stacking). They differ along several dimensions:
//!
//! | Concern                | per-element-push          | collect-then-bulk        |
//! | ---------------------- | ------------------------- | ------------------------ |
//! | Leaf storage           | `Vec<#native>` / MBVA / bitmap | `Vec<&T>` (+ `Vec<Option<IdxSize>>`) |
//! | Per-elem push          | typed buffer push          | `&v` ref push (+ scatter)  |
//! | Offsets-buf own policy | Move (single-use)          | Clone (shared across arms) |
//! | Freeze placement       | interleaved per layer      | hoisted above branch dispatch |
//! | Dispatch arms          | 1 (always direct)          | 2 (no-IO) or 4 (with-IO)    |
//! | Output cardinality     | one `Series` push          | for-loop over inner schema   |
//! | Inner-option layers    | unbounded                  | `<= 1` (debug-asserted)      |
//!
//! [`LeafKind`] captures these differences behind one type. The unified
//! emitter [`super::emit::vec_emit_general`] dispatches on `LeafKind` to
//! shape the storage decls, per-row push, post-scan materialization, and
//! per-layer wrap-policy.
//!
//! Bool-d1-bare carve-out: depth-1 `Vec<bool>` with no inner Option and no
//! outer Option layers takes a bespoke path in [`super::vec`] (flat
//! `Vec<bool>` + `BooleanArray::from_slice`, faster than bit-packing for
//! the all-non-null case). The carve-out lives outside `LeafKind` and
//! bypasses `vec_emit_general` entirely.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::VecLayers;

use super::idents;
use super::shape_walk::{LayerIdents, LayerWrap, OwnPolicy};

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
    ///
    /// This is a residual perf-driven choice: hoisting the freezes for the
    /// per-element-push path reproducibly regresses depth-N benches 4-12%
    /// (see comment in [`super::vec`]); per-element-push interleaves the
    /// freeze with each wrap. Conversely, interleaving for the
    /// collect-then-bulk path would re-freeze the same offsets buffer per
    /// inner-schema-column iteration of every dispatch arm — wasteful and
    /// also slower, so the nested path hoists once.
    pub(super) const fn freeze_hoisted(&self) -> bool {
        matches!(self, Self::CollectThenBulk(_))
    }

    /// Outer-some-prefix used by the scan walker for this leaf kind.
    /// `__df_derive_some_` for per-element-push (matches the historical
    /// flat-vec path); `__df_derive_n_some_` for collect-then-bulk.
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
    /// `__df_derive_n_some_` for scan) for byte-equivalence with the legacy
    /// emission, where precount and scan emitted under different prefix
    /// conventions historically.
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

/// Per-emitter "shape × leaf-kind" intermediates the unified emitter passes
/// down to materialization. Captures the per-layer ident bundle and the
/// `VecLayers` so leaf-kind-specific post-scan emission helpers don't have to
/// recompute them.
pub(super) struct EmitShape<'a> {
    pub shape: &'a VecLayers,
    pub layers: &'a [LayerIdents],
}

impl<'a> EmitShape<'a> {
    pub(super) const fn new(shape: &'a VecLayers, layers: &'a [LayerIdents]) -> Self {
        Self { shape, layers }
    }

    pub(super) const fn depth(&self) -> usize {
        self.shape.depth()
    }

    /// Build the per-layer `LayerWrap` slice the shared list-stack helper
    /// consumes. Each layer's `freeze_decl` is empty for the
    /// collect-then-bulk path (freezes hoisted above) and contains the
    /// `OffsetsBuffer::try_from(...)?` plus optional `Bitmap::from(...)`
    /// for the per-element-push path (freezes interleaved with each wrap).
    pub(super) fn layer_wraps(
        &self,
        kind: &LeafKind<'_>,
        pa_root: &TokenStream,
    ) -> Vec<LayerWrap<'_>> {
        let mut out: Vec<LayerWrap<'_>> = Vec::with_capacity(self.depth());
        for (cur, layer) in self.layers.iter().enumerate() {
            let buf_id = &layer.offsets_buf;
            let validity_bm = if self.shape.layers[cur].has_outer_validity() {
                Some(&layer.validity_bm)
            } else {
                None
            };
            let freeze_decl = if kind.freeze_hoisted() {
                TokenStream::new()
            } else {
                let offsets = &layer.offsets;
                let mut fd = quote! {
                    let #buf_id: #pa_root::offset::OffsetsBuffer<i64> =
                        #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
                };
                if let Some(bm_id) = validity_bm {
                    let validity_mb = &layer.validity_mb;
                    fd.extend(quote! {
                        let #bm_id: #pa_root::bitmap::Bitmap =
                            <#pa_root::bitmap::Bitmap as ::core::convert::From<
                                #pa_root::bitmap::MutableBitmap,
                            >>::from(#validity_mb);
                    });
                }
                fd
            };
            out.push(LayerWrap {
                offsets_buf: kind.layer_own_policy(buf_id),
                validity_bm,
                freeze_decl,
            });
        }
        out
    }

    /// Build the hoisted-freeze pair for the collect-then-bulk path:
    /// converts each layer's `MutableBitmap` to `Bitmap` (where the layer
    /// has an outer Option) and each layer's `Vec<i64>` to
    /// `OffsetsBuffer<i64>`. Returns empty token streams for the
    /// per-element-push path (freezes interleaved per-layer instead).
    ///
    /// The pair is `(validity_freeze, offsets_freeze)` — the call site
    /// emits validity first (any outer-Option arms reference the frozen
    /// `Bitmap` lifetime) then offsets at the head of each branch.
    pub(super) fn hoisted_freezes(
        &self,
        kind: &LeafKind<'_>,
        pa_root: &TokenStream,
    ) -> (TokenStream, TokenStream) {
        if !kind.freeze_hoisted() {
            return (TokenStream::new(), TokenStream::new());
        }
        let mut validity_freeze: Vec<TokenStream> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if !self.shape.layers[i].has_outer_validity() {
                continue;
            }
            let mb = &layer.validity_mb;
            let bm = &layer.validity_bm;
            validity_freeze.push(quote! {
                let #bm: #pa_root::bitmap::Bitmap =
                    <#pa_root::bitmap::Bitmap as ::core::convert::From<
                        #pa_root::bitmap::MutableBitmap,
                    >>::from(#mb);
            });
        }
        let mut offsets_freeze: Vec<TokenStream> = Vec::new();
        for layer in self.layers {
            let offsets = &layer.offsets;
            let buf = &layer.offsets_buf;
            offsets_freeze.push(quote! {
                let #buf: #pa_root::offset::OffsetsBuffer<i64> =
                    #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
            });
        }
        (
            quote! { #(#validity_freeze)* },
            quote! { #(#offsets_freeze)* },
        )
    }
}
