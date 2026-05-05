//! Encoder IR: a compositional encoder model for per-field `DataFrame` columnization.
//!
//! Each leaf encoder knows how to emit (decls, push, finish) for one base type.
//! The `option(inner)` combinator wraps a leaf to add `Option<...>` semantics.
//! `vec(inner)` adds an arbitrary-depth `LargeListArray` stack over the leaf.
//! Per-field codegen folds the wrapper stack right-to-left over the leaf to
//! assemble the final emission.
//!
//! Each `LeafSpec` carries two arms — `bare` for the unwrapped shape and
//! `option` for the `[Option]` shape. The split lets the `bool` leaf
//! override the option case with a 3-arm match (so `Some(false)` is a true
//! no-op against a values bitmap pre-filled with `false`) and lets each
//! leaf pick the right buffer/finish layout for its option semantics
//! without leaking a runtime "is-this-supplied?" check into the dispatcher.
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

pub(in crate::codegen) mod idents;
mod leaf;
mod nested;
mod option;
mod shape_walk;
mod vec;

use crate::ir::{BaseType, DateTimeUnit, PrimitiveTransform, Wrapper};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub use nested::{NestedLeafCtx, build_nested_encoder};

/// Build the type-as-path token stream for a struct/generic field. For a
/// struct referenced without args (e.g. `Address`) this is the bare ident;
/// for a struct referenced with args (e.g. `Foo<M>` or `Foo<M, N>`) it is
/// the turbofish form `Foo::<M, N>`, valid in both expression and type
/// position. Generic type parameters use the bare ident (the macro injects
/// the trait bounds that make `T::method()` resolve).
pub fn build_type_path(
    ident: &Ident,
    args: Option<&syn::AngleBracketedGenericArguments>,
) -> TokenStream {
    args.map_or_else(
        || quote! { #ident },
        |ab| {
            let inner = &ab.args;
            quote! { #ident::<#inner> }
        },
    )
}

/// Narrowed enum of every (base, transform) combination the parser can
/// construct that reaches the encoder. Strategy splits at `build_field_emit`:
/// `Struct`/`Generic` bases without a stringy transform route through
/// `build_nested_encoder`; everything else lands here. The variants are the
/// canonical primitive leaf shapes, with the parser-coupled fields (`DateTime`
/// unit, `Decimal` precision/scale, stringy base ident) inlined so downstream
/// dispatch is a single exhaustive match — no `_ => unreachable!()` arms.
pub(super) enum LeafShape<'a> {
    /// `i8`/`i16`/`i32`/`i64`/`u8`/`u16`/`u32`/`u64`/`f32`/`f64`.
    /// The borrowed `BaseType` retrieves `NumericInfo` downstream.
    Numeric(&'a BaseType),
    /// `ISize`/`USize` widened to `i64`/`u64` at the leaf push site.
    NumericWidened(&'a BaseType),
    /// Bare `String` field (no transform).
    String,
    /// Bare `bool` field (no transform).
    Bool,
    /// `chrono::DateTime<Utc>` paired with the parser-injected
    /// `DateTimeToInt(unit)` transform.
    DateTime(DateTimeUnit),
    /// `rust_decimal::Decimal` paired with the parser-injected
    /// `DecimalToInt128 { precision, scale }` transform.
    Decimal { precision: u8, scale: u8 },
    /// `to_string` transform (any `Display` base materializes to `String`).
    AsString,
    /// `as_str` borrow path. Base is restricted to types implementing
    /// `AsRef<str>`; the parser rejects every other base.
    AsStr(StringyBase<'a>),
}

/// Bases that can be paired with the `as_str` transform: `String`,
/// concrete user struct types, and generic type parameters. The parser's
/// `derive_transform` enforces this set.
pub(super) enum StringyBase<'a> {
    String,
    Struct {
        ident: &'a Ident,
        args: Option<&'a syn::AngleBracketedGenericArguments>,
    },
    Generic(&'a Ident),
}

impl<'a> LeafShape<'a> {
    /// Project a parser-validated `(base, transform)` pair onto a narrowed
    /// `LeafShape`. The single panic site for "encoder reached with a base
    /// the parser cannot construct" lives here — every other dispatcher
    /// downstream consumes `LeafShape` and is exhaustively typed.
    pub(super) fn from_base_transform(
        base: &'a BaseType,
        transform: Option<&'a PrimitiveTransform>,
    ) -> Self {
        match transform {
            None => match base {
                BaseType::I8
                | BaseType::I16
                | BaseType::I32
                | BaseType::I64
                | BaseType::U8
                | BaseType::U16
                | BaseType::U32
                | BaseType::U64
                | BaseType::F32
                | BaseType::F64 => Self::Numeric(base),
                BaseType::ISize | BaseType::USize => Self::NumericWidened(base),
                BaseType::String => Self::String,
                BaseType::Bool => Self::Bool,
                BaseType::DateTimeUtc
                | BaseType::Decimal
                | BaseType::Struct(..)
                | BaseType::Generic(_) => unreachable!(
                    "df-derive: encoder boundary reached with no-transform \
                     DateTime/Decimal/Struct/Generic base — \
                     parser injects transforms for DateTime/Decimal and \
                     routes Struct/Generic through the nested encoder"
                ),
            },
            Some(PrimitiveTransform::DateTimeToInt(unit)) => match base {
                BaseType::DateTimeUtc => Self::DateTime(*unit),
                BaseType::F64
                | BaseType::F32
                | BaseType::I64
                | BaseType::U64
                | BaseType::I32
                | BaseType::U32
                | BaseType::I16
                | BaseType::U16
                | BaseType::I8
                | BaseType::U8
                | BaseType::Bool
                | BaseType::String
                | BaseType::ISize
                | BaseType::USize
                | BaseType::Decimal
                | BaseType::Struct(..)
                | BaseType::Generic(_) => unreachable!(
                    "df-derive: DateTimeToInt transform paired with non-DateTime \
                     base — parser injects this transform only for DateTime<Utc>"
                ),
            },
            Some(PrimitiveTransform::DecimalToInt128 { precision, scale }) => match base {
                BaseType::Decimal => Self::Decimal {
                    precision: *precision,
                    scale: *scale,
                },
                BaseType::F64
                | BaseType::F32
                | BaseType::I64
                | BaseType::U64
                | BaseType::I32
                | BaseType::U32
                | BaseType::I16
                | BaseType::U16
                | BaseType::I8
                | BaseType::U8
                | BaseType::Bool
                | BaseType::String
                | BaseType::ISize
                | BaseType::USize
                | BaseType::DateTimeUtc
                | BaseType::Struct(..)
                | BaseType::Generic(_) => unreachable!(
                    "df-derive: DecimalToInt128 transform paired with non-Decimal \
                     base — parser injects this transform only for rust_decimal::Decimal"
                ),
            },
            Some(PrimitiveTransform::ToString) => Self::AsString,
            Some(PrimitiveTransform::AsStr) => Self::AsStr(StringyBase::from_base(base)),
        }
    }
}

impl<'a> StringyBase<'a> {
    /// Project a parser-validated `as_str` base onto `StringyBase`. The
    /// parser's `derive_transform` guarantees only
    /// `String`/`Struct`/`Generic` reach this projection.
    fn from_base(base: &'a BaseType) -> Self {
        match base {
            BaseType::String => Self::String,
            BaseType::Struct(ident, args) => Self::Struct {
                ident,
                args: args.as_ref(),
            },
            BaseType::Generic(ident) => Self::Generic(ident),
            BaseType::F64
            | BaseType::F32
            | BaseType::I64
            | BaseType::U64
            | BaseType::I32
            | BaseType::U32
            | BaseType::I16
            | BaseType::U16
            | BaseType::I8
            | BaseType::U8
            | BaseType::Bool
            | BaseType::ISize
            | BaseType::USize
            | BaseType::DateTimeUtc
            | BaseType::Decimal => unreachable!(
                "df-derive: as_str transform paired with non-stringy base — \
                 parser rejects every base except String/Struct/Generic"
            ),
        }
    }

    /// Token stream for the type-as-path expression used in UFCS calls
    /// (`<#ty as AsRef<str>>::as_ref(...)`).
    pub(super) fn ty_path(&self) -> TokenStream {
        match self {
            Self::String => quote! { ::std::string::String },
            Self::Struct { ident, args } => build_type_path(ident, *args),
            Self::Generic(ident) => quote! { #ident },
        }
    }

    /// True when the stringy base is bare `String`. The bare-`String` path
    /// has tighter token shapes (deref-coercion through `&String`) than the
    /// UFCS path.
    pub(super) const fn is_string(&self) -> bool {
        matches!(self, Self::String)
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
    let param = idents::collapse_option_param();
    let mut out = quote! { (#base).as_ref() };
    for _ in 1..n {
        out = quote! { #out.and_then(|#param| #param.as_ref()) };
    }
    out
}

/// Per-field encoder state. Variant determines whether the encoder serves
/// the single-column primitive path (`Leaf`) or the multi-column nested
/// path (`Multi`); the field set is type-enforced per variant.
pub enum Encoder {
    /// Single-column primitive leaf path. `decls` is emitted once before the
    /// per-row loop; `push` is spliced inside the loop; `series` is an
    /// expression that evaluates to a `polars::prelude::Series` after the
    /// loop, which the caller wraps as `columns.push(series.into())`.
    Leaf {
        decls: Vec<TokenStream>,
        push: TokenStream,
        series: TokenStream,
    },
    /// Multi-column nested path. The `columnar` block executes once after
    /// the call site's per-row loop and pushes one Series per inner schema
    /// column of the nested type onto the outer `columns` vec directly.
    Multi { columnar: TokenStream },
}

/// Per-leaf identity shared across both the primitive ([`LeafCtx`]) and
/// nested ([`NestedLeafCtx`]) encoder paths: the per-row access expression,
/// the field's stable index (used to namespace generated idents), and the
/// field's column name.
pub struct BaseCtx<'a> {
    pub access: &'a TokenStream,
    pub idx: usize,
    pub name: &'a str,
}

/// Per-leaf metadata threaded into the leaf builders.
pub struct LeafCtx<'a> {
    pub base: BaseCtx<'a>,
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
    let shape = LeafShape::from_base_transform(base, transform);
    match normalize_wrappers(wrappers) {
        WrapperKind::Leaf { option_layers: 0 } => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(&shape, ctx).bare;
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        WrapperKind::Leaf { option_layers: 1 } => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(&shape, ctx).option;
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        // Primitive multi-Option leaf shapes (`Option<Option<T>>`,
        // `Option<Option<Option<T>>>`, …): pre-collapse the access to
        // `Option<&T>` (Polars folds every nested None to one validity bit)
        // and feed it through the single-Option leaf machinery via a
        // synthesized per-row local. Nested multi-Option (`Option<Option<T>>`
        // over a struct/generic) is handled separately in `build_nested_encoder`.
        WrapperKind::Leaf {
            option_layers: layers,
        } => option::wrap_multi_option_primitive(&shape, ctx, layers),
        WrapperKind::Vec(vec_shape) => vec::try_build_vec_encoder(&shape, ctx, &vec_shape),
    }
}
