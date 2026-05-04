//! Encoder IR: a compositional encoder model for per-field `DataFrame` columnization.
//!
//! Each leaf encoder knows how to emit (decls, push, finish) for one base type.
//! The `option(inner)` combinator wraps a leaf to add `Option<...>` semantics.
//! `vec(inner)` adds an arbitrary-depth `LargeListArray` stack over the leaf.
//! Per-field codegen folds the wrapper stack right-to-left over the leaf to
//! assemble the final emission.
//!
//! Each leaf carries two push token streams: `bare_push` for the unwrapped
//! shape, and `option_push` for the `[Option]` shape. The split lets the
//! `bool` leaf override the option case with a 3-arm match (so `Some(false)`
//! is a true no-op against a values bitmap pre-filled with `false`).
//!
//! `Vec`-bearing wrappers are normalized into a [`VecShape`] (one entry per
//! `Vec` layer; each entry tracks whether an outer `Option` adjoins it as
//! list-level validity). After normalization the encoder emits an N-deep
//! precount, an N-deep push loop, an N-deep stack of `LargeListArray::new`
//! calls — one flat values buffer at the deepest layer, one optional inner
//! validity bitmap at the deepest layer, one optional outer-list validity
//! bitmap per `Vec` layer. Polars folds consecutive `Option` layers into a
//! single validity bit, so `[Option, Option]` collapses to `[Option]` and
//! `[Option, Option, Vec]` collapses to `[Option, Vec]` before the encoder
//! sees them — the runtime semantics match because the only observable null
//! is whichever bit is the outermost Polars validity.

mod leaf;
mod nested;
mod option;
mod shape_walk;
mod vec;

use crate::ir::{BaseType, PrimitiveTransform, Wrapper};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

pub use nested::{NestedLeafCtx, build_nested_encoder};

/// Per-field identifier convention for the encoder IR's primitive leaves.
/// Funneling every declaration site through this struct turns rename
/// mistakes into a compile error at the helper itself.
///
/// Nested-struct/generic encoders manage their own per-shape ident
/// bundles (`NestedIdents` / `NestedLayerIdents`).
struct PopulatorIdents;

impl PopulatorIdents {
    /// Owning `Vec<T>` / `Vec<Option<T>>` buffer for a primitive scalar
    /// field. Holds `Vec<&str>` / `Vec<Option<&str>>` on the borrowing fast
    /// path.
    fn primitive_buf(idx: usize) -> Ident {
        format_ident!("__df_derive_buf_{}", idx)
    }

    /// `MutableBitmap` validity buffer for the
    /// `is_direct_primitive_array_option_numeric_leaf` fast path. Paired
    /// with `primitive_buf` (which holds `Vec<#native>` on that path) so the
    /// finisher can build a `PrimitiveArray::new(dtype, vals, validity)`
    /// directly without a `Vec<Option<T>>` second walk.
    fn primitive_validity(idx: usize) -> Ident {
        format_ident!("__df_derive_val_{}", idx)
    }

    /// Row counter for the `is_direct_view_option_string_leaf` fast path.
    /// Indexes the pre-filled `MutableBitmap` so the per-row push only
    /// writes a single byte for `None` rows via `set_unchecked`, instead
    /// of pushing both `true` and `false` bits unconditionally.
    fn primitive_row_idx(idx: usize) -> Ident {
        format_ident!("__df_derive_ri_{}", idx)
    }

    /// Reused `String` scratch buffer for the
    /// `is_direct_view_to_string_leaf` fast path. Paired with `primitive_buf`
    /// (which holds `MutableBinaryViewArray<str>` on that path) so each row
    /// can clear-and-write into the scratch via `Display::fmt` and then push
    /// the resulting `&str` into the view array (which copies the bytes),
    /// avoiding a fresh per-row `String` allocation.
    fn primitive_str_scratch(idx: usize) -> Ident {
        format_ident!("__df_derive_str_{}", idx)
    }
}

/// One `Vec` layer in a normalized wrapper stack. Outermost layer first.
#[derive(Clone, Copy, Debug)]
struct VecLayer {
    /// Number of consecutive `Option` wrappers immediately above this `Vec`.
    /// Zero means no list-level validity. `>0` means a `MutableBitmap`
    /// rides under this `LargeListArray` and the per-row access expression
    /// must walk `option_layers` `Option`s before entering the `Vec`.
    /// Polars only carries one validity bit per list level, so all
    /// consecutive `Option`s collapse to one bit (`Some(None)` and `None`
    /// are indistinguishable in the column).
    option_layers: usize,
}

impl VecLayer {
    const fn has_outer_validity(self) -> bool {
        self.option_layers > 0
    }
}

/// Normalized form of a `Vec`-bearing wrapper stack. `layers[0]` is the
/// outermost `Vec`. `inner_option_layers` is the count of consecutive
/// `Option` wrappers immediately surrounding the leaf (between the
/// innermost `Vec` and the leaf type). `>0` means a per-element validity
/// bit is stored at the leaf and the per-element access expression must
/// walk `inner_option_layers` `Option`s before reaching the leaf value.
#[derive(Clone, Debug)]
struct VecShape {
    layers: Vec<VecLayer>,
    inner_option_layers: usize,
}

impl VecShape {
    const fn depth(&self) -> usize {
        self.layers.len()
    }

    fn any_outer_validity(&self) -> bool {
        self.layers.iter().any(|l| l.has_outer_validity())
    }

    const fn has_inner_option(&self) -> bool {
        self.inner_option_layers > 0
    }
}

/// Normalize a wrapper stack into either a leaf shape (`[]` or `[Option]+`)
/// or a `VecShape`.
///
/// `WrapperKind::Leaf { option_layers }` covers no-`Vec` shapes. The count
/// is the number of `Option`s applied (zero for a bare leaf, `>0` for any
/// `Option<…<Option<T>>>` stack). Consecutive `Option`s all fold into a
/// single validity bit — Polars cannot represent two distinct null states.
///
/// `WrapperKind::Vec(VecShape)` covers any shape with at least one `Vec`.
/// Each layer records how many `Option`s sit immediately above it (folded
/// into list-level validity); `inner_option_layers` covers the trailing
/// `Option`s above the leaf (folded into per-element validity).
enum WrapperKind {
    Leaf { option_layers: usize },
    Vec(VecShape),
}

fn normalize_wrappers(wrappers: &[Wrapper]) -> WrapperKind {
    let mut layers: Vec<VecLayer> = Vec::new();
    let mut pending_options: usize = 0;
    let mut inner_option_layers: usize = 0;
    let mut saw_vec = false;
    for w in wrappers {
        match w {
            Wrapper::Option => {
                if saw_vec {
                    inner_option_layers += 1;
                } else {
                    pending_options += 1;
                }
            }
            Wrapper::Vec => {
                saw_vec = true;
                // Options accumulated since the last Vec wrap THIS Vec from
                // the previous Vec's element perspective: from the new Vec's
                // POV they sit immediately above it as list-level validity.
                // Drop them into the new layer instead of discarding.
                layers.push(VecLayer {
                    option_layers: pending_options + std::mem::take(&mut inner_option_layers),
                });
                pending_options = 0;
            }
        }
    }
    if layers.is_empty() {
        return WrapperKind::Leaf {
            option_layers: pending_options,
        };
    }
    VecShape {
        layers,
        inner_option_layers,
    }
    .into()
}

impl From<VecShape> for WrapperKind {
    fn from(s: VecShape) -> Self {
        Self::Vec(s)
    }
}

/// Build an expression that collapses `n` `Option` layers above a base
/// expression into a single `Option<&Inner>`. `base` must already be a
/// reference (or place expression that auto-derefs to a reference). For
/// `n == 0` this is a no-op (returns the base unchanged).
fn collapse_options_to_ref(base: &TokenStream, n: usize) -> TokenStream {
    if n == 0 {
        return base.clone();
    }
    let mut out = quote! { (#base).as_ref() };
    for _ in 1..n {
        out = quote! { #out.and_then(|__df_derive_o| __df_derive_o.as_ref()) };
    }
    out
}

/// How a leaf consumes values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    /// Leaf consumes one value at a time via a `push` token stream.
    PerElementPush,
    /// Leaf collects refs across all rows then performs one bulk encode call.
    /// Used for nested-struct/generic base types that route through
    /// `<T as Columnar>::columnar_from_refs(&refs)`.
    #[allow(dead_code)]
    CollectThenBulk,
}

/// What an `Encoder`'s finish step produces.
///
/// Primitive leaves materialize a single `Series` (the field's column).
/// Nested-struct/generic leaves materialize **multiple** Series, one per
/// inner schema column of the nested type — so they are emitted as a block
/// that pushes directly onto the call site's `columns` vec.
pub enum EncoderFinish {
    /// Single-Series finish: an expression that evaluates to a
    /// `polars::prelude::Series`. Outer call sites wrap as
    /// `columns.push(s.into())`.
    Series(TokenStream),
    /// Multi-column finish: a pre-built block that pushes one Series per
    /// inner schema column onto the call site's `columns` vec.
    Multi { columnar: TokenStream },
}

/// Per-field encoder state. `decls` and `finish` are emitted once at the
/// top/bottom of the columnar populator pipeline; `push` is spliced inside
/// the per-row loop.
pub struct Encoder {
    pub decls: Vec<TokenStream>,
    /// Push tokens used when this encoder is the top of the wrapper stack.
    pub push: TokenStream,
    /// Push tokens specifically for an outer `option(...)` wrapper. `None`
    /// makes the option combinator generate a generic 2-arm match.
    pub option_push: Option<TokenStream>,
    pub finish: EncoderFinish,
    pub kind: LeafKind,
    /// 0 for leaves, +1 per `vec` layer. Used by Step 2.
    #[allow(dead_code)]
    pub offset_depth: usize,
}

impl Encoder {
    /// Convenience: wrap a `Series`-valued token expression as `Encoder.finish`.
    const fn series_finish(expr: TokenStream) -> EncoderFinish {
        EncoderFinish::Series(expr)
    }

    /// Consume the encoder and extract its `EncoderFinish::Series` payload.
    /// Panics if the encoder produces multi-column output — primitive call
    /// sites only ever build single-Series encoders, so this is invariant.
    pub fn into_series_finish(self) -> TokenStream {
        match self.finish {
            EncoderFinish::Series(ts) => ts,
            EncoderFinish::Multi { .. } => {
                unreachable!("into_series_finish called on a multi-column encoder")
            }
        }
    }
}

/// Per-leaf metadata threaded into the leaf builders.
pub struct LeafCtx<'a> {
    pub access: &'a TokenStream,
    pub idx: usize,
    pub name: &'a str,
    pub decimal128_encode_trait: &'a TokenStream,
}

// --- Top-level dispatcher ---

/// Returns `Some(encoder)` when the (base, transform, wrappers) triple can be
/// served by this encoder IR. Covers every wrapper stack the parser accepts
/// for every primitive leaf — bare numerics, `ISize`/`USize` (widened to
/// `i64`/`u64` at the codegen boundary), `String`, `Bool`, `Decimal`,
/// `DateTime`, `as_str`, `to_string`, plus `Option<…<Option<T>>>` stacks of
/// arbitrary depth (Polars folds consecutive `Option`s into a single
/// validity bit, so the encoder collapses the access expression to a
/// single `Option<&T>` before invoking the option-leaf push) and every
/// primitive vec-bearing shape. The function is total on parser-validated
/// IR; combinations the parser cannot construct panic in `build_leaf`.
pub fn build_encoder(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    ctx: &LeafCtx<'_>,
) -> Encoder {
    match normalize_wrappers(wrappers) {
        WrapperKind::Leaf { option_layers: 0 } => vec::build_leaf(base, transform, ctx),
        WrapperKind::Leaf { option_layers: 1 } => {
            let leaf = vec::build_leaf(base, transform, ctx);
            option::wrap_option(base, transform, leaf, ctx)
        }
        // Primitive multi-Option leaf shapes (`Option<Option<T>>`,
        // `Option<Option<Option<T>>>`, …): pre-collapse the access to
        // `Option<&T>` (Polars folds every nested None to one validity bit)
        // and feed it through the single-Option leaf machinery via a
        // synthesized per-row local. Nested multi-Option (`Option<Option<T>>`
        // over a struct/generic) is handled separately in `build_nested_encoder`.
        WrapperKind::Leaf {
            option_layers: layers,
        } => option::wrap_multi_option_primitive(base, transform, ctx, layers),
        WrapperKind::Vec(shape) => vec::try_build_vec_encoder(base, transform, ctx, &shape),
    }
}
