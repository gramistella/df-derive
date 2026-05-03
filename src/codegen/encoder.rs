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

use crate::ir::{BaseType, DateTimeUnit, PrimitiveTransform, Wrapper};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::populator_idents::PopulatorIdents;

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
                inner_option_layers = 0;
                layers.push(VecLayer {
                    option_layers: pending_options,
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

// --- Common decl helpers ---

/// `let mut #buf: Vec<#elem> = Vec::with_capacity(items.len());`
fn vec_decl(buf: &syn::Ident, elem: &TokenStream) -> TokenStream {
    quote! {
        let mut #buf: ::std::vec::Vec<#elem> =
            ::std::vec::Vec::with_capacity(items.len());
    }
}

/// `let mut #ident: MutableBitmap = MutableBitmap::with_capacity(items.len());`
/// (no pre-fill — push-based use only).
fn mb_decl(ident: &syn::Ident) -> TokenStream {
    let pa_root = super::polars_paths::polars_arrow_root();
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
    }
}

/// `let mut #ident: MutableBitmap = MutableBitmap pre-filled with #value over items.len();`
fn mb_decl_filled(ident: &syn::Ident, value: bool) -> TokenStream {
    let pa_root = super::polars_paths::polars_arrow_root();
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b = #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
            __df_derive_b.extend_constant(items.len(), #value);
            __df_derive_b
        };
    }
}

/// `let mut #ident: usize = 0;` — row counter for the pre-filled-bitmap leaves.
fn row_idx_decl(ident: &syn::Ident) -> TokenStream {
    quote! { let mut #ident: usize = 0; }
}

/// `let mut #buf: MutableBinaryViewArray<str> = MutableBinaryViewArray::<str>::with_capacity(items.len());`
fn mbva_decl(buf: &syn::Ident) -> TokenStream {
    let pa_root = super::polars_paths::polars_arrow_root();
    quote! {
        let mut #buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(items.len());
    }
}

/// Convert a `MutableBitmap` validity buffer into the `Option<Bitmap>`
/// `with_chunk` / `with_validity` arms expect. `MutableBitmap -> Option<Bitmap>`
/// collapses to `None` when no bits are unset, preserving the no-null fast path.
fn validity_into_option(validity: &syn::Ident) -> TokenStream {
    let pa_root = super::polars_paths::polars_arrow_root();
    quote! {
        ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
            #validity,
        )
    }
}

/// Build a Series via `into_series(StringChunked::with_chunk(name, arr))`.
fn string_chunked_series(name: &str, arr_expr: &TokenStream) -> TokenStream {
    let pp = super::polars_paths::prelude();
    quote! {
        #pp::IntoSeries::into_series(
            #pp::StringChunked::with_chunk(#name.into(), { #arr_expr }),
        )
    }
}

// --- Leaf builders ---

/// Bare numeric primitive leaf (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64`).
/// Uses `Vec<#native>` storage. The bare finisher swaps `Series::new(&Vec)`
/// for `<Chunked>::from_vec(name, buf)`, which consumes the Vec without
/// copying. The option combinator switches to `PrimitiveArray::new` over
/// the `Vec<#native>` + parallel `MutableBitmap`.
fn numeric_leaf(ctx: &LeafCtx<'_>, base: &BaseType) -> Encoder {
    let info = super::type_registry::numeric_info(base)
        .expect("numeric_leaf must be called with a numeric BaseType");
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let native = info.native;
    let chunked = info.chunked;
    let access = ctx.access;
    let name = ctx.name;
    let pp = super::polars_paths::prelude();

    let bare_push = quote! { #buf.push((#access).clone()); };
    let finish_series = quote! {
        #pp::IntoSeries::into_series(#chunked::from_vec(#name.into(), #buf))
    };
    // `Some` arm pushes the value (validity pre-filled to `true` is wrong —
    // we use push-based MutableBitmap here, no pre-fill); `None` arm pushes
    // `<#native>::default()` and `validity.push(false)`. Splitting value vs
    // validity into independent pushes lets the compiler vectorize cleanly.
    let option_push = quote! {
        match #access {
            ::std::option::Option::Some(__df_derive_v) => {
                #buf.push(__df_derive_v);
                #validity.push(true);
            }
            ::std::option::Option::None => {
                #buf.push(<#native as ::std::default::Default>::default());
                #validity.push(false);
            }
        }
    };
    Encoder {
        decls: vec![vec_decl(&buf, &native)],
        push: bare_push,
        option_push: Some(option_push),
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// Option combinator for numeric leaves: adds a parallel `MutableBitmap`
/// validity buffer and finishes through `PrimitiveArray::new` +
/// `<Chunked>::with_chunk`.
fn option_for_numeric(ctx: &LeafCtx<'_>, base: &BaseType, inner: Encoder) -> Encoder {
    let info = super::type_registry::numeric_info(base)
        .expect("option_for_numeric requires a numeric BaseType");
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let pa_root = super::polars_paths::polars_arrow_root();
    let native = info.native;
    let chunked = info.chunked;
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    let valid_opt = validity_into_option(&validity);
    let finish_series = quote! {{
        let arr = #pa_root::array::PrimitiveArray::<#native>::new(
            <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
            #buf.into(),
            #valid_opt,
        );
        #pp::IntoSeries::into_series(#chunked::with_chunk(#name.into(), arr))
    }};
    Encoder {
        decls: vec![vec_decl(&buf, &native), mb_decl(&validity)],
        push: inner.option_push.unwrap_or(inner.push),
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

/// Bare `String` leaf — accumulates straight into a
/// `MutableBinaryViewArray<str>` buffer. Bypasses the `Vec<&str>` round-trip
/// and the second walk `Series::new(&Vec<&str>)` would do via `from_slice_values`.
fn string_leaf(ctx: &LeafCtx<'_>) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let row_idx = PopulatorIdents::primitive_row_idx(ctx.idx);
    let access = ctx.access;
    let name = ctx.name;

    let bare_push = quote! { #buf.push_value_ignore_validity((#access).as_str()); };
    let finish_series = string_chunked_series(name, &quote! { #buf.freeze() });
    // Option arm: split-buffer pair (MBVA + MutableBitmap pre-filled to
    // `true`). `Some` pushes the borrowed `&str` (no validity work); `None`
    // pushes "" and flips a single bit via the safe `MutableBitmap::set`.
    let option_push = quote! {
        match &(#access) {
            ::std::option::Option::Some(__df_derive_v) => {
                #buf.push_value_ignore_validity(__df_derive_v.as_str());
            }
            ::std::option::Option::None => {
                #buf.push_value_ignore_validity("");
                #validity.set(#row_idx, false);
            }
        }
        #row_idx += 1;
    };
    Encoder {
        decls: vec![mbva_decl(&buf)],
        push: bare_push,
        option_push: Some(option_push),
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// Option combinator for the `String` / `as_string` MBVA-based leaves.
/// Adds the `MutableBitmap` validity buffer + row counter and finishes by
/// freezing the values into a `Utf8ViewArray`, attaching the bitmap via
/// `with_validity`, and wrapping in `StringChunked::with_chunk`.
fn option_for_string_like(
    ctx: &LeafCtx<'_>,
    extra_decls: Vec<TokenStream>,
    inner: Encoder,
) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let row_idx = PopulatorIdents::primitive_row_idx(ctx.idx);
    let name = ctx.name;
    let valid_opt = validity_into_option(&validity);
    let mut decls = vec![mbva_decl(&buf)];
    decls.extend(extra_decls);
    decls.push(mb_decl_filled(&validity, true));
    decls.push(row_idx_decl(&row_idx));
    let arr_expr = quote! { #buf.freeze().with_validity(#valid_opt) };
    Encoder {
        decls,
        push: inner
            .option_push
            .expect("string-like leaf must supply option_push"),
        option_push: None,
        finish: Encoder::series_finish(string_chunked_series(name, &arr_expr)),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

/// Bare `bool` leaf — `Vec<bool>` + `Series::new`. Keeps the slow path because
/// `BooleanChunked::from_slice` is bulk and faster than `BooleanArray::new` +
/// `with_chunk` for the all-non-null case.
fn bool_leaf(ctx: &LeafCtx<'_>) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let row_idx = PopulatorIdents::primitive_row_idx(ctx.idx);
    let access = ctx.access;
    let name = ctx.name;
    let pp = super::polars_paths::prelude();

    let bare_push = quote! { #buf.push((#access).clone()); };
    let finish_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    // 3-arm form for the option case. The `option_for_bool` combinator
    // switches `decls` to the bitmap-pair layout so `#buf` is now a
    // `MutableBitmap`, not a `Vec<bool>`. `Some(true)` flips a value bit,
    // `Some(false)` is zero work (values pre-filled to `false`), `None`
    // flips a validity bit (validity pre-filled to `true`).
    let option_push = quote! {
        match (#access) {
            ::std::option::Option::Some(true) => { #buf.set(#row_idx, true); }
            ::std::option::Option::Some(false) => {}
            ::std::option::Option::None => { #validity.set(#row_idx, false); }
        }
        #row_idx += 1;
    };
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { bool })],
        push: bare_push,
        option_push: Some(option_push),
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// Option combinator for the bool leaf: bit-packed values bitmap (pre-filled
/// `false`) + validity bitmap (pre-filled `true`) + row counter. Finishes
/// through `BooleanArray::new` + `BooleanChunked::with_chunk`.
fn option_for_bool(ctx: &LeafCtx<'_>, inner: Encoder) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let row_idx = PopulatorIdents::primitive_row_idx(ctx.idx);
    let pa_root = super::polars_paths::polars_arrow_root();
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    let valid_opt = validity_into_option(&validity);
    let finish_series = quote! {{
        let arr = #pa_root::array::BooleanArray::new(
            #pa_root::datatypes::ArrowDataType::Boolean,
            ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#buf),
            #valid_opt,
        );
        #pp::IntoSeries::into_series(
            #pp::BooleanChunked::with_chunk(#name.into(), arr),
        )
    }};
    Encoder {
        decls: vec![
            mb_decl_filled(&buf, false),
            mb_decl_filled(&validity, true),
            row_idx_decl(&row_idx),
        ],
        push: inner
            .option_push
            .expect("bool_leaf must supply option_push"),
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

/// Build push tokens for a `Vec<...>` (or `Vec<Option<...>>`) buffer that
/// holds the result of a per-row mapped expression — used by `Decimal` and
/// `DateTime` leaves which share the same shape (`buf.push({ mapped })` for
/// bare, `match Some/None => Some(mapped)/None` for option).
fn mapped_push_pair(
    ctx: &LeafCtx<'_>,
    transform: &PrimitiveTransform,
) -> (TokenStream, TokenStream) {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let access = ctx.access;
    let decimal_trait = ctx.decimal128_encode_trait;
    let mapped_bare =
        super::type_registry::map_primitive_expr(access, Some(transform), decimal_trait);
    let mapped_some = {
        let some_var = quote! { __df_derive_v };
        super::type_registry::map_primitive_expr(&some_var, Some(transform), decimal_trait)
    };
    let bare_push = quote! { #buf.push({ #mapped_bare }); };
    let option_push = quote! {
        match &(#access) {
            ::std::option::Option::Some(__df_derive_v) => {
                #buf.push(::std::option::Option::Some({ #mapped_some }));
            }
            ::std::option::Option::None => {
                #buf.push(::std::option::Option::None);
            }
        }
    };
    (bare_push, option_push)
}

/// `Decimal` leaf with a `DecimalToInt128` transform. Bare: `Vec<i128>` +
/// `Int128Chunked::from_vec` + `into_decimal_unchecked`. Option: switches to
/// `Vec<Option<i128>>` + `from_iter_options` + `into_decimal_unchecked`.
fn decimal_leaf(ctx: &LeafCtx<'_>, precision: u8, scale: u8) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    let int128 = super::polars_paths::int128_chunked();
    let p = precision as usize;
    let s = scale as usize;
    let transform = PrimitiveTransform::DecimalToInt128 { precision, scale };
    let (bare_push, option_push) = mapped_push_pair(ctx, &transform);
    let finish_series = quote! {{
        let ca = #int128::from_vec(#name.into(), #buf);
        #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
    }};
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { i128 })],
        push: bare_push,
        option_push: Some(option_push),
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// Option combinator for Decimal: switches to `Vec<Option<i128>>` and
/// `from_iter_options`-based finish.
fn option_for_decimal(ctx: &LeafCtx<'_>, precision: u8, scale: u8, inner: Encoder) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    let int128 = super::polars_paths::int128_chunked();
    let p = precision as usize;
    let s = scale as usize;
    let finish_series = quote! {{
        let ca = <#int128 as #pp::NewChunkedArray<_, _>>::from_iter_options(
            #name.into(),
            #buf.into_iter(),
        );
        #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
    }};
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i128> })],
        push: inner
            .option_push
            .expect("decimal_leaf must supply option_push"),
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

/// `DateTime<Utc>` leaf with a `DateTimeToInt(unit)` transform. Bare:
/// `Vec<i64>` + `Series::new` + cast to `Datetime(unit, None)`. Option:
/// switches to `Vec<Option<i64>>` with the same finish path.
fn datetime_leaf(ctx: &LeafCtx<'_>, unit: DateTimeUnit) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    let transform = PrimitiveTransform::DateTimeToInt(unit);
    let (bare_push, option_push) = mapped_push_pair(ctx, &transform);
    let dtype =
        super::type_registry::compute_mapping(&BaseType::DateTimeUtc, Some(&transform), &[])
            .full_dtype;
    let finish_series = quote! {{
        let mut s = <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf);
        s = s.cast(&#dtype)?;
        s
    }};
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { i64 })],
        push: bare_push,
        option_push: Some(option_push),
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// Option combinator for `DateTime`: switches to `Vec<Option<i64>>`. The
/// finish path is identical structurally (`Series::new` + cast); only the
/// element type changes — so we can reuse the inner finish.
fn option_for_datetime(ctx: &LeafCtx<'_>, unit: DateTimeUnit, inner: Encoder) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    let dtype = super::type_registry::compute_mapping(
        &BaseType::DateTimeUtc,
        Some(&PrimitiveTransform::DateTimeToInt(unit)),
        &[],
    )
    .full_dtype;
    let finish_series = quote! {{
        let mut s = <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf);
        s = s.cast(&#dtype)?;
        s
    }};
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i64> })],
        push: inner
            .option_push
            .expect("datetime_leaf must supply option_push"),
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

/// `as_string` (Display) leaf. Reused `String` scratch + MBVA accumulator —
/// each row clears the scratch, runs `Display::fmt` into it, then pushes the
/// resulting `&str` to the view array (which copies the bytes).
fn as_string_leaf(ctx: &LeafCtx<'_>) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let scratch = PopulatorIdents::primitive_str_scratch(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let row_idx = PopulatorIdents::primitive_row_idx(ctx.idx);
    let access = ctx.access;
    let name = ctx.name;

    let bare_push = quote! {
        {
            use ::std::fmt::Write as _;
            #scratch.clear();
            ::std::write!(&mut #scratch, "{}", &(#access)).unwrap();
            #buf.push_value_ignore_validity(#scratch.as_str());
        }
    };
    let finish_series = string_chunked_series(name, &quote! { #buf.freeze() });
    let option_push = quote! {
        match &(#access) {
            ::std::option::Option::Some(__df_derive_v) => {
                use ::std::fmt::Write as _;
                #scratch.clear();
                ::std::write!(&mut #scratch, "{}", __df_derive_v).unwrap();
                #buf.push_value_ignore_validity(#scratch.as_str());
            }
            ::std::option::Option::None => {
                #buf.push_value_ignore_validity("");
                #validity.set(#row_idx, false);
            }
        }
        #row_idx += 1;
    };
    Encoder {
        decls: vec![
            mbva_decl(&buf),
            quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); },
        ],
        push: bare_push,
        option_push: Some(option_push),
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// `as_str` (borrowed) leaf. `Vec<&str>` (or `Vec<Option<&str>>` in option
/// context) borrows from `items`. `ty_path` is the type-path expression for
/// UFCS — `String`, the field's struct ident, or a generic-parameter ident.
fn as_str_leaf(ctx: &LeafCtx<'_>, ty_path: &TokenStream, base: &BaseType) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let access = ctx.access;
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    // Bare `String` base (with redundant `as_str`) and `as_str` on a
    // non-string base have different push expressions: `String`'s `&String`
    // deref-coerces to `&str` so the plain `&` form works there; non-string
    // bases need UFCS through the type path. Match `classify_borrow`'s split.
    let is_string = matches!(base, BaseType::String);
    let bare_push = if is_string {
        quote! { #buf.push(&(#access)); }
    } else {
        quote! { #buf.push(<#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#access))); }
    };
    let option_push = if is_string {
        quote! { #buf.push((#access).as_deref()); }
    } else {
        quote! {
            #buf.push(
                (#access).as_ref().map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
            );
        }
    };
    let finish_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { &str })],
        push: bare_push,
        option_push: Some(option_push),
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// Option combinator for `as_str`: switches to `Vec<Option<&str>>` storage.
/// The finish path is `Series::new(name, &buf)` regardless.
fn option_for_as_str(ctx: &LeafCtx<'_>, inner: Encoder) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = super::polars_paths::prelude();
    let finish_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<&str> })],
        push: inner
            .option_push
            .expect("as_str_leaf must supply option_push"),
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

// --- Vec combinator (depth 1 and 2) ---

/// Per-leaf-kind plumbing the `vec_for_*` combinator needs to assemble a
/// flat `Vec<Native>` / `MutableBinaryViewArray<str>` / `MutableBitmap`
/// over all elements across all outer rows, then wrap it in one or two
/// `LargeListArray::new` layers. Each variant pairs a per-element store
/// expression (used by both bare and option-Some arms) with the dtype
/// tokens needed to build the leaf array and route through the in-scope
/// `__df_derive_assemble_list_series_unchecked` helper.
///
/// Per-leaf-kind classification used by the `vec(inner)` combinator. The
/// per-element push and the leaf-array build differ across these variants;
/// every other piece of the bulk emit (precount loops, offsets vecs, list
/// stacking, `__df_derive_assemble_list_series_unchecked` routing) is shared.
///
/// `Bool` carries no extra fields because the bit-packed values bitmap and
/// the no-validity / pre-filled-validity arrays are constructed the same way
/// regardless of context — the only knob is `has_inner_option`. Bare bool
/// (`Vec<bool>` / `Vec<Vec<bool>>`) follows a different layout entirely
/// (flat `Vec<bool>` for depth 1, bit-packed bitmap for depth 2) and is
/// served by a dedicated builder, not this enum.
enum VecLeafSpec {
    /// Numeric (i8/i16/.../f64) and the single-`i128`/`i64` transform leaves
    /// (`Decimal` -> `i128`, `DateTime` -> `i64`). All share
    /// `PrimitiveArray::new` over `Vec<Native>` plus an optional
    /// inner-validity bitmap.
    ///
    /// `value_expr` materializes the leaf value from the binding
    /// `__df_derive_v` — the loop binding is `&T` for bare shapes and the
    /// `Some(v)` arm of the inner option-match for nullable shapes (same
    /// binding name in both, so a single expression suffices).
    Numeric {
        native: TokenStream,
        value_expr: TokenStream,
        /// Adds an anonymous `use #decimal128_encode_trait as _;` so the
        /// dot-syntax `try_to_i128_mantissa` resolves on `&Decimal`.
        needs_decimal_import: bool,
    },
    /// `String` / `to_string` / `as_str` over `MutableBinaryViewArray<str>`.
    /// `value_expr` materializes a `&str` from the `__df_derive_v` binding
    /// (same convention as `Numeric`). `extra_decls` covers the per-row
    /// `String` scratch used by `to_string` (placed before the MBVA so
    /// cleanup ordering matches prior emission).
    StringLike {
        value_expr: TokenStream,
        extra_decls: Vec<TokenStream>,
    },
    /// Inner-Option bool — bit-packed values bitmap (pre-filled `false`)
    /// plus a parallel validity bitmap (pre-filled `true`). Bare bool is
    /// served separately because it gets a flat `Vec<bool>` at depth 1.
    Bool,
    /// Bare bool — bit-packed values bitmap (pre-filled `false`), no
    /// inner-Option. Used for depth >= 2 (and any shape that has an outer
    /// validity layer above the bool leaf, since the depth-1 fast path
    /// requires both no inner option and no outer option).
    BoolBare,
}

/// Bool-specific helper: returns the leaf-array construction tokens given
/// whether the parent shape carries an inner validity bitmap.
fn bool_leaf_array_tokens(
    pa_root: &TokenStream,
    has_inner_option: bool,
    values_ident: &syn::Ident,
    validity_ident: &syn::Ident,
) -> TokenStream {
    if has_inner_option {
        let valid_opt = validity_into_option(validity_ident);
        quote! {
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#values_ident),
                #valid_opt,
            )
        }
    } else {
        quote! {
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#values_ident),
                ::std::option::Option::None,
            )
        }
    }
}

/// Per-`Vec` layer ident set. `offsets` accumulates per-list element counts
/// for this layer; `validity` is the optional list-level `MutableBitmap`
/// (only when `has_outer_validity`). Layer `i` is the `i`-th `Vec` from the
/// outside; layer `depth-1` is the innermost (its `offsets` track flat-leaf
/// counts; deeper layers' `offsets` track child-list counts).
struct VecLayerIdents {
    offsets: syn::Ident,
    validity: syn::Ident,
    /// Per-layer iteration binding. Layer 0 binds the field access; deeper
    /// layers bind the previous layer's iterator output.
    bind: syn::Ident,
}

fn vec_layer_idents(depth: usize) -> Vec<VecLayerIdents> {
    (0..depth)
        .map(|i| VecLayerIdents {
            offsets: format_ident!("__df_derive_layer_off_{}", i),
            validity: format_ident!("__df_derive_layer_val_{}", i),
            bind: format_ident!("__df_derive_layer_bind_{}", i),
        })
        .collect()
}

/// Build the entire `vec(inner)` (or deeper) emit block for a normalized
/// [`VecShape`]. Emits a single `let __df_derive_field_series_<idx> = { ... };`
/// declaration the caller splices into the populator's pre-loop decls.
///
/// The bulk-fusion contract: regardless of depth, the leaf storage (flat
/// values buffer + optional validity bitmap) is allocated and populated
/// once; the layer stack adds one `LargeListArray::new` per `Vec` wrapper
/// and one `MutableBitmap` per layer that has an outer-`Option`.
fn vec_emit_decl(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecShape,
    leaf_dtype_tokens: &TokenStream,
) -> TokenStream {
    let pa_root = super::polars_paths::polars_arrow_root();
    let pp = super::polars_paths::prelude();
    let access = ctx.access;
    let series_local = vec_encoder_series_local(ctx.idx);
    let leaf_bind = format_ident!("__df_derive_v");
    let layers = vec_layer_idents(shape.depth());

    let (precount_decls, leaf_capacity_expr) = vec_precount_pieces(access, shape, &layers);
    let (leaf_storage_decls, per_elem_push, leaf_arr_expr) = build_vec_leaf_pieces(
        spec,
        shape.has_inner_option(),
        &leaf_capacity_expr,
        &pa_root,
    );

    // Decimal mantissa rescale dispatches through the `Decimal128Encode`
    // trait via dot syntax — the trait must be in scope so method
    // resolution finds it. Anonymous `use ... as _;` keeps the user's
    // namespace clean. Other `Numeric` variants and `StringLike` / `Bool`
    // don't reference any user trait.
    let extra_imports = if let VecLeafSpec::Numeric {
        needs_decimal_import: true,
        ..
    } = spec
    {
        let trait_path = ctx.decimal128_encode_trait;
        quote! { use #trait_path as _; }
    } else {
        TokenStream::new()
    };

    let leaf_offsets_post_push = leaf_offsets_post_push_tokens(spec);
    let offsets_decls = vec_offsets_decls(shape, &layers);
    let validity_decls = vec_layer_validity_decls(shape, &layers, &pa_root);
    let push_loops = vec_push_loops(
        access,
        shape,
        &layers,
        &leaf_bind,
        &per_elem_push,
        &leaf_offsets_post_push,
    );
    let final_assemble = vec_final_assemble(shape, &layers, leaf_dtype_tokens, &pa_root, &pp);

    quote! {
        let #series_local: #pp::Series = {
            #extra_imports
            #precount_decls
            #leaf_storage_decls
            #offsets_decls
            #validity_decls
            #push_loops
            #leaf_arr_expr
            #final_assemble
        };
    }
}

/// Precount loop + leaf-capacity expression for the depth-N bulk-vec emit.
/// Returns `(decls, leaf_capacity)` — the leaf capacity is the running total
/// of leaf elements summed across every nested `Vec` layer of every outer row.
///
/// Mirrors the structure of `vec_push_loops` so the precount and the actual
/// push loop walk the same `Some/None` arms in lock-step. Layers with
/// `has_outer_validity` skip both the layer-counter increment and the
/// recursion on `None`, matching the runtime push logic that records a
/// repeat-offset for the null cell.
#[allow(clippy::items_after_statements)]
fn vec_precount_pieces(
    access: &TokenStream,
    shape: &VecShape,
    layers: &[VecLayerIdents],
) -> (TokenStream, TokenStream) {
    let depth = shape.depth();
    let total_leaves = format_ident!("__df_derive_total_leaves");
    let layer_counters: Vec<syn::Ident> = (0..depth.saturating_sub(1))
        .map(|i| format_ident!("__df_derive_total_layer_{}", i))
        .collect();

    fn build_iter_body(
        shape: &VecShape,
        layers: &[VecLayerIdents],
        layer_counters: &[syn::Ident],
        total_leaves: &syn::Ident,
        cur: usize,
        vec_bind: &TokenStream,
    ) -> TokenStream {
        let depth = shape.depth();
        if cur + 1 == depth {
            quote! { #total_leaves += #vec_bind.len(); }
        } else {
            let inner_bind = &layers[cur + 1].bind;
            let counter = &layer_counters[cur];
            let inner_layer_body = build_layer_body(
                shape,
                layers,
                layer_counters,
                total_leaves,
                cur + 1,
                &quote! { #inner_bind },
            );
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_layer_body
                    #counter += 1;
                }
            }
        }
    }

    fn build_layer_body(
        shape: &VecShape,
        layers: &[VecLayerIdents],
        layer_counters: &[syn::Ident],
        total_leaves: &syn::Ident,
        cur: usize,
        bind: &TokenStream,
    ) -> TokenStream {
        let opt_layers = shape.layers[cur].option_layers;
        if opt_layers > 0 {
            let inner_vec_bind = format_ident!("__df_derive_some_{}", cur);
            let inner_iter = build_iter_body(
                shape,
                layers,
                layer_counters,
                total_leaves,
                cur,
                &quote! { #inner_vec_bind },
            );
            // Collapse N consecutive Options to one match. Polars semantics:
            // every nested None collapses to the same null. The collapsed
            // expression evaluates to `Option<&Vec<...>>`. For the
            // `opt_layers == 1` case, default binding modes match
            // `&Option<Vec<...>>` directly without an explicit `.as_ref()`,
            // which LLVM doesn't always eliminate.
            let collapsed = if opt_layers == 1 {
                bind.clone()
            } else {
                collapse_options_to_ref(bind, opt_layers)
            };
            quote! {
                if let ::std::option::Option::Some(#inner_vec_bind) = #collapsed {
                    #inner_iter
                }
            }
        } else {
            build_iter_body(shape, layers, layer_counters, total_leaves, cur, bind)
        }
    }

    let layer0_iter_src = quote! { (&(#access)) };
    let body = build_layer_body(
        shape,
        layers,
        &layer_counters,
        &total_leaves,
        0,
        &layer0_iter_src,
    );
    let counter_decls = layer_counters
        .iter()
        .map(|c| quote! { let mut #c: usize = 0; });
    let pre = quote! {
        let mut #total_leaves: usize = 0;
        #(#counter_decls)*
        for __df_derive_it in items {
            #body
        }
    };
    (pre, quote! { #total_leaves })
}

/// Build the nested for-loop push body for the depth-N vec emit.
///
/// At each layer `i`:
/// - if `has_outer_validity`: the source binding is an `Option<Vec<...>>`. A
///   `None` arm pushes validity=false and skips iteration (the layer's
///   offset just repeats the previous value, so the outer cell is treated
///   as "null with no children"). A `Some` arm pushes validity=true and
///   iterates the inner Vec normally.
/// - if not: the source binding is a `Vec<...>`; iterate it directly.
/// - after the iteration arm, push this layer's offset (next inner layer's
///   offsets length minus 1, or for the innermost layer the leaf-buffer
///   length).
#[allow(clippy::items_after_statements, clippy::too_many_lines)]
fn vec_push_loops(
    access: &TokenStream,
    shape: &VecShape,
    layers: &[VecLayerIdents],
    leaf_bind: &syn::Ident,
    per_elem_push: &TokenStream,
    leaf_offsets_post_push: &TokenStream,
) -> TokenStream {
    fn build_inner_iter(
        shape: &VecShape,
        layers: &[VecLayerIdents],
        leaf_bind: &syn::Ident,
        per_elem_push: &TokenStream,
        leaf_offsets_post_push: &TokenStream,
        cur: usize,
        // Expression bound to the inner Vec at this layer (after Option-
        // unwrap when has_outer_validity, otherwise the layer's source bind).
        vec_bind: &TokenStream,
    ) -> TokenStream {
        let depth = shape.depth();
        if cur + 1 == depth {
            // At the deepest Vec, iterate elements. The per_elem_push body
            // expects `__df_derive_v` to be either:
            // - `&T` directly (no inner-Option), bound by the for-loop, or
            // - `Option<&T>` (inner-Option), with the per-elem push then
            //   matching it. To support `inner_option_layers > 1`, we
            //   collapse the for-loop binding through `as_ref().and_then`
            //   into a single `Option<&T>` before splicing the push body.
            //   The bare for-loop binding is `__df_derive_v_raw` and the
            //   collapsed one becomes `__df_derive_v`.
            if shape.has_inner_option() {
                if shape.inner_option_layers == 1 {
                    quote! {
                        for #leaf_bind in #vec_bind.iter() {
                            #per_elem_push
                        }
                    }
                } else {
                    let raw_bind = format_ident!("__df_derive_v_raw");
                    let collapsed =
                        collapse_options_to_ref(&quote! { #raw_bind }, shape.inner_option_layers);
                    quote! {
                        for #raw_bind in #vec_bind.iter() {
                            let #leaf_bind: ::std::option::Option<_> = #collapsed;
                            #per_elem_push
                        }
                    }
                }
            } else {
                quote! {
                    for #leaf_bind in #vec_bind.iter() {
                        #per_elem_push
                    }
                }
            }
        } else {
            let inner_bind = &layers[cur + 1].bind;
            let inner_layer_body = build_layer_body(
                shape,
                layers,
                leaf_bind,
                per_elem_push,
                leaf_offsets_post_push,
                cur + 1,
                &quote! { #inner_bind },
            );
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_layer_body
                }
            }
        }
    }

    fn build_layer_body(
        shape: &VecShape,
        layers: &[VecLayerIdents],
        leaf_bind: &syn::Ident,
        per_elem_push: &TokenStream,
        leaf_offsets_post_push: &TokenStream,
        cur: usize,
        bind: &TokenStream,
    ) -> TokenStream {
        let depth = shape.depth();
        let layer = &layers[cur];
        let offsets = &layer.offsets;
        let offsets_post_push = if cur + 1 == depth {
            leaf_offsets_post_push.clone()
        } else {
            let inner_offsets = &layers[cur + 1].offsets;
            quote! { (#inner_offsets.len() - 1) }
        };
        let opt_layers = shape.layers[cur].option_layers;
        let inner_iter = if opt_layers > 0 {
            // The bind here holds `&Option<...<Option<Vec<...>>>>` with
            // `opt_layers` of nesting. Collapse to `Option<&Vec<...>>` and
            // match: Some(v) pushes validity=true and iterates v; None
            // pushes validity=false and skips. Polars folds every nested
            // None into the same null.
            let validity = &layer.validity;
            let inner_vec_bind = format_ident!("__df_derive_some_{}", cur);
            let inner_iter = build_inner_iter(
                shape,
                layers,
                leaf_bind,
                per_elem_push,
                leaf_offsets_post_push,
                cur,
                &quote! { #inner_vec_bind },
            );
            let collapsed = if opt_layers == 1 {
                bind.clone()
            } else {
                collapse_options_to_ref(bind, opt_layers)
            };
            quote! {
                match #collapsed {
                    ::std::option::Option::Some(#inner_vec_bind) => {
                        #validity.push(true);
                        #inner_iter
                    }
                    ::std::option::Option::None => {
                        #validity.push(false);
                    }
                }
            }
        } else {
            build_inner_iter(
                shape,
                layers,
                leaf_bind,
                per_elem_push,
                leaf_offsets_post_push,
                cur,
                bind,
            )
        };
        quote! {
            #inner_iter
            #offsets.push(#offsets_post_push as i64);
        }
    }

    let layer0_iter_src = quote! { (&(#access)) };
    let body = build_layer_body(
        shape,
        layers,
        leaf_bind,
        per_elem_push,
        leaf_offsets_post_push,
        0,
        &layer0_iter_src,
    );
    quote! {
        for __df_derive_it in items {
            #body
        }
    }
}

/// Per-layer offsets vec declarations. Layer 0 is sized `items.len() + 1`;
/// deeper layers use the precounted layer-N counter when available
/// (`__df_derive_total_layer_{N-1} + 1`).
fn vec_offsets_decls(shape: &VecShape, layers: &[VecLayerIdents]) -> TokenStream {
    let depth = shape.depth();
    let mut out: Vec<TokenStream> = Vec::with_capacity(depth);
    for (i, layer) in layers.iter().enumerate() {
        let offsets = &layer.offsets;
        let cap = if i == 0 {
            quote! { items.len() + 1 }
        } else {
            let counter = format_ident!("__df_derive_total_layer_{}", i - 1);
            quote! { #counter + 1 }
        };
        out.push(quote! {
            let mut #offsets: ::std::vec::Vec<i64> =
                ::std::vec::Vec::with_capacity(#cap);
            #offsets.push(0);
        });
    }
    quote! { #(#out)* }
}

/// Per-layer outer-`Option` validity bitmap declarations. Allocated push-
/// based (no pre-fill) — `Some` arms push `true`, `None` arms push `false`.
fn vec_layer_validity_decls(
    shape: &VecShape,
    layers: &[VecLayerIdents],
    pa_root: &TokenStream,
) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if !shape.layers[i].has_outer_validity() {
            continue;
        }
        let validity = &layer.validity;
        let cap = if i == 0 {
            quote! { items.len() }
        } else {
            let counter = format_ident!("__df_derive_total_layer_{}", i - 1);
            quote! { #counter }
        };
        out.push(quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(#cap);
        });
    }
    quote! { #(#out)* }
}

/// Stack `depth` `LargeListArray::new` layers and route the outermost one
/// through `__df_derive_assemble_list_series_unchecked`. Each layer's
/// validity (when its `Option` is present) is folded onto its
/// `LargeListArray`. The helper wraps the supplied logical dtype in one
/// `List<>` layer; for depth N we pre-wrap `leaf_dtype_tokens` in
/// `N - 1` extra `List<>` envelopes so the schema dtype matches the
/// runtime list nesting.
fn vec_final_assemble(
    shape: &VecShape,
    layers: &[VecLayerIdents],
    leaf_dtype_tokens: &TokenStream,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let depth = shape.depth();
    let mut block: Vec<TokenStream> = Vec::new();
    // Layer indexing convention: layer `depth - 1` is the innermost (wraps
    // the leaf array). Layer 0 is the outermost (passed to the helper).
    let mut prev_arr = format_ident!("__df_derive_leaf_arr");
    for cur in (0..depth).rev() {
        let layer = &layers[cur];
        let offsets_buf_id = format_ident!("__df_derive_layer_off_buf_{}", cur);
        let arr_id = format_ident!("__df_derive_list_arr_{}", cur);
        let validity_expr = if shape.layers[cur].has_outer_validity() {
            let validity = &layer.validity;
            quote! {
                ::std::option::Option::Some(
                    <#pa_root::bitmap::Bitmap as ::core::convert::From<
                        #pa_root::bitmap::MutableBitmap,
                    >>::from(#validity)
                )
            }
        } else {
            quote! { ::std::option::Option::None }
        };
        let offsets = &layer.offsets;
        let prev_arr_local = prev_arr.clone();
        block.push(quote! {
            let #offsets_buf_id: #pa_root::offset::OffsetsBuffer<i64> =
                #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
            let #arr_id: #pp::LargeListArray = #pp::LargeListArray::new(
                #pp::LargeListArray::default_datatype(
                    #pa_root::array::Array::dtype(&#prev_arr_local).clone(),
                ),
                #offsets_buf_id,
                ::std::boxed::Box::new(#prev_arr_local) as #pp::ArrayRef,
                #validity_expr,
            );
        });
        prev_arr = arr_id;
    }
    // Wrap the leaf logical dtype in `(depth - 1)` extra `List<>` layers.
    // The helper wraps once more, yielding `List<List<...List<leaf>>>`
    // with `depth` total `List<>` envelopes.
    let mut helper_logical = leaf_dtype_tokens.clone();
    for _ in 0..depth.saturating_sub(1) {
        helper_logical = quote! { #pp::DataType::List(::std::boxed::Box::new(#helper_logical)) };
    }
    let outer_arr = prev_arr;
    quote! {
        #(#block)*
        __df_derive_assemble_list_series_unchecked(
            #outer_arr,
            #helper_logical,
        )
    }
}

/// The expression that becomes `<offsets>.push(<expr> as i64)` at the
/// inner-vec-loop tail. Numeric / decimal / datetime use the flat-vec
/// length; the string-view path uses the MBVA `len()` (which is the count
/// of pushed views); the bool bit-packed-bitmap layout tracks a per-leaf
/// index so it can stay in sync with both the values and validity bitmaps.
fn leaf_offsets_post_push_tokens(spec: &VecLeafSpec) -> TokenStream {
    match spec {
        VecLeafSpec::Numeric { .. } => quote! { __df_derive_flat.len() },
        VecLeafSpec::StringLike { .. } => quote! { __df_derive_view_buf.len() },
        VecLeafSpec::Bool | VecLeafSpec::BoolBare => quote! { __df_derive_leaf_idx },
    }
}

/// Per-leaf storage decls + per-element push + leaf-array build.
///
/// Returns `(leaf_storage_decls, per_elem_push, leaf_arr_expr)`.
/// `leaf_storage_decls` covers the values buffer + (when `has_inner_option`)
/// the parallel `MutableBitmap` validity. `per_elem_push` is one per-element
/// `match` (option) or push (bare) referencing the binding `__df_derive_v`.
/// `leaf_arr_expr` produces the typed `PrimitiveArray<...>` /
/// `Utf8ViewArray` / `BooleanArray` value bound to `__df_derive_leaf_arr`.
fn build_vec_leaf_pieces(
    spec: &VecLeafSpec,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    match spec {
        VecLeafSpec::Numeric {
            native,
            value_expr,
            needs_decimal_import: _,
        } => numeric_leaf_pieces(
            native,
            value_expr,
            has_inner_option,
            leaf_capacity_expr,
            pa_root,
        ),
        VecLeafSpec::StringLike {
            value_expr,
            extra_decls,
        } => string_like_leaf_pieces(
            value_expr,
            extra_decls,
            has_inner_option,
            leaf_capacity_expr,
            pa_root,
        ),
        VecLeafSpec::Bool => {
            debug_assert!(
                has_inner_option,
                "VecLeafSpec::Bool only handles the inner-Option case",
            );
            bool_inner_option_leaf_pieces(leaf_capacity_expr, pa_root)
        }
        VecLeafSpec::BoolBare => {
            debug_assert!(
                !has_inner_option,
                "VecLeafSpec::BoolBare only handles the no-inner-Option case",
            );
            bool_bare_leaf_pieces(leaf_capacity_expr, pa_root)
        }
    }
}

/// Bit-packed values bitmap pre-filled `false`, with a per-leaf index
/// counter for `set(idx, true)`. Used by deeper-than-1 / outer-Option-bearing
/// `Vec<bool>` shapes that don't qualify for the depth-1 `from_slice`
/// fast path. The leaf-array build skips the validity argument.
fn bool_bare_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let storage = quote! {
        let mut __df_derive_values: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            __df_derive_b.extend_constant(#leaf_capacity_expr, false);
            __df_derive_b
        };
        let mut __df_derive_leaf_idx: usize = 0;
    };
    let push = quote! {
        if *__df_derive_v {
            __df_derive_values.set(__df_derive_leaf_idx, true);
        }
        __df_derive_leaf_idx += 1;
    };
    let values_ident = format_ident!("__df_derive_values");
    let validity_ident = format_ident!("__df_derive_validity");
    let leaf_arr_inner = bool_leaf_array_tokens(pa_root, false, &values_ident, &validity_ident);
    let leaf_arr_expr = quote! {
        let __df_derive_leaf_arr: #pa_root::array::BooleanArray = #leaf_arr_inner;
    };
    (storage, push, leaf_arr_expr)
}

fn numeric_leaf_pieces(
    native: &TokenStream,
    value_expr: &TokenStream,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    // Numeric / Decimal / DateTime: `Vec<#native>` flat values.
    // Inner-Option carries a parallel `MutableBitmap` pre-filled `true`.
    let storage = if has_inner_option {
        quote! {
            let mut __df_derive_flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
            let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
                let mut __df_derive_b =
                    #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
                __df_derive_b.extend_constant(#leaf_capacity_expr, true);
                __df_derive_b
            };
        }
    } else {
        quote! {
            let mut __df_derive_flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
        }
    };
    // The bare and inner-Option arms both reference `__df_derive_v` — the
    // bare arm gets it from the loop binding directly (the for-loop in
    // `vec_emit_decl` binds the leaf as `__df_derive_v`), the option arm
    // gets it from the `Some(v)` pattern. Sharing one `value_expr` avoids
    // two near-duplicate per-spec expressions.
    let push = if has_inner_option {
        quote! {
            match __df_derive_v {
                ::std::option::Option::Some(__df_derive_v) => {
                    __df_derive_flat.push(#value_expr);
                }
                ::std::option::Option::None => {
                    __df_derive_flat.push(<#native as ::std::default::Default>::default());
                    __df_derive_validity.set(__df_derive_flat.len() - 1, false);
                }
            }
        }
    } else {
        quote! {
            __df_derive_flat.push(#value_expr);
        }
    };
    let leaf_arr_expr = if has_inner_option {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::new(
                    <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                    __df_derive_flat.into(),
                    ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                        __df_derive_validity,
                    ),
                );
        }
    } else {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::from_vec(__df_derive_flat);
        }
    };
    (storage, push, leaf_arr_expr)
}

fn string_like_leaf_pieces(
    value_expr: &TokenStream,
    extra_decls: &[TokenStream],
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    // String / to_string / as_str: `MutableBinaryViewArray<str>` flat values.
    // Inner-Option uses a separate validity bitmap pre-filled `true`, plus
    // a row index that advances per element so `validity.set(i, false)`
    // hits the right bit on `None`.
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    for d in extra_decls {
        storage_parts.push(d.clone());
    }
    storage_parts.push(quote! {
        let mut __df_derive_view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        storage_parts.push(quote! {
            let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
                let mut __df_derive_b =
                    #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
                __df_derive_b.extend_constant(#leaf_capacity_expr, true);
                __df_derive_b
            };
            let mut __df_derive_leaf_idx: usize = 0;
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let push = if has_inner_option {
        quote! {
            match __df_derive_v {
                ::std::option::Option::Some(__df_derive_v) => {
                    __df_derive_view_buf.push_value_ignore_validity({ #value_expr });
                }
                ::std::option::Option::None => {
                    __df_derive_view_buf.push_value_ignore_validity("");
                    __df_derive_validity.set(__df_derive_leaf_idx, false);
                }
            }
            __df_derive_leaf_idx += 1;
        }
    } else {
        quote! {
            __df_derive_view_buf.push_value_ignore_validity({ #value_expr });
        }
    };
    let leaf_arr_expr = if has_inner_option {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::Utf8ViewArray = __df_derive_view_buf
                .freeze()
                .with_validity(
                    ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                        __df_derive_validity,
                    ),
                );
        }
    } else {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::Utf8ViewArray = __df_derive_view_buf.freeze();
        }
    };
    (storage, push, leaf_arr_expr)
}

fn bool_inner_option_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    // Bool with inner-Option only — bare bool is served by
    // `vec_encoder_bool_bare`. Bit-packed values bitmap (pre-filled `false`)
    // + parallel validity bitmap (pre-filled `true`) + leaf index counter so
    // `set(i, ...)` lands on the right bit.
    let storage = quote! {
        let mut __df_derive_values: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            __df_derive_b.extend_constant(#leaf_capacity_expr, false);
            __df_derive_b
        };
        let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            __df_derive_b.extend_constant(#leaf_capacity_expr, true);
            __df_derive_b
        };
        let mut __df_derive_leaf_idx: usize = 0;
    };
    let push = quote! {
        match __df_derive_v {
            ::std::option::Option::Some(true) => {
                __df_derive_values.set(__df_derive_leaf_idx, true);
            }
            ::std::option::Option::Some(false) => {}
            ::std::option::Option::None => {
                __df_derive_validity.set(__df_derive_leaf_idx, false);
            }
        }
        __df_derive_leaf_idx += 1;
    };
    let values_ident = format_ident!("__df_derive_values");
    let validity_ident = format_ident!("__df_derive_validity");
    let leaf_arr_inner = bool_leaf_array_tokens(pa_root, true, &values_ident, &validity_ident);
    let leaf_arr_expr = quote! {
        let __df_derive_leaf_arr: #pa_root::array::BooleanArray = #leaf_arr_inner;
    };
    (storage, push, leaf_arr_expr)
}

/// Per-field local for the assembled Series — one per (field, depth)
/// combination, namespaced by `idx` so two adjacent fields don't collide.
fn vec_encoder_series_local(idx: usize) -> syn::Ident {
    format_ident!("__df_derive_field_series_{}", idx)
}

/// Build the encoder for a `[Vec, ...]` shape: a single `decls` statement
/// declaring `let series_local = { ... };`, an empty `push`, and a
/// finish that just references the local (with the columnar-context rename
/// applied via `with_name(name)` — the vec-anyvalues context passes
/// `name = ""`, so the rename is a no-op there).
fn vec_encoder(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecShape,
    leaf_dtype: &TokenStream,
) -> Encoder {
    let series_local = vec_encoder_series_local(ctx.idx);
    let decl = vec_emit_decl(ctx, spec, shape, leaf_dtype);
    // The decl binds `series_local` to the assembled Series. The caller
    // wraps the encoder's finish expression in `with_name(...)` /
    // `AnyValue::List(...)` as appropriate.
    let finish_series = quote! { #series_local };
    Encoder {
        decls: vec![decl],
        push: TokenStream::new(),
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: shape.depth(),
    }
}

/// Bare-bool variant of the vec encoder. At depth 1 with no inner-Option and
/// no outer-Option layers, uses `BooleanArray::from_slice` (bulk, no
/// bit-packing). For deeper or option-bearing shapes, routes through the
/// generalized `vec_encoder` with `VecLeafSpec::BoolBare` (a bit-packed
/// `MutableBitmap` set per element). The depth-1 fast path matches the
/// legacy `from_slice`-based emission byte-for-byte.
fn vec_encoder_bool_bare(ctx: &LeafCtx<'_>, shape: &VecShape) -> Encoder {
    if shape.depth() == 1 && !shape.any_outer_validity() {
        let pa_root = super::polars_paths::polars_arrow_root();
        let pp = super::polars_paths::prelude();
        let series_local = vec_encoder_series_local(ctx.idx);
        let body = bool_bare_depth1_body(ctx.access, &pa_root, &pp);
        let name = ctx.name;
        let decl = quote! { let #series_local: #pp::Series = { #body }; };
        return Encoder {
            decls: vec![decl],
            push: TokenStream::new(),
            option_push: None,
            finish: Encoder::series_finish(quote! { #series_local.with_name(#name.into()) }),
            kind: LeafKind::PerElementPush,
            offset_depth: 1,
        };
    }
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    vec_encoder(ctx, &VecLeafSpec::BoolBare, shape, &leaf_dtype)
}

/// `Vec<bool>` body: `Vec::extend` per outer row into a flat `Vec<bool>`,
/// then `BooleanArray::from_slice` at the end. No bit-packing because
/// `from_slice` is bulk and faster than `set` for the all-non-null case.
fn bool_bare_depth1_body(
    access: &TokenStream,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let inner_offsets = format_ident!("__df_derive_inner_offsets");
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    quote! {
        let mut __df_derive_total_leaves: usize = 0;
        for __df_derive_it in items {
            __df_derive_total_leaves += (&(#access)).len();
        }
        let mut __df_derive_flat: ::std::vec::Vec<bool> =
            ::std::vec::Vec::with_capacity(__df_derive_total_leaves);
        let mut #inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #inner_offsets.push(0);
        for __df_derive_it in items {
            __df_derive_flat.extend((&(#access)).iter().copied());
            #inner_offsets.push(__df_derive_flat.len() as i64);
        }
        let __df_derive_leaf_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::from_slice(&__df_derive_flat);
        let __df_derive_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#inner_offsets)?;
        let __df_derive_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_leaf_arr).clone(),
            ),
            __df_derive_offsets_buf,
            ::std::boxed::Box::new(__df_derive_leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_list_arr,
            #leaf_dtype,
        )
    }
}

/// `Vec<...>` (`as_str` transform) — same MBVA-based encoder as the bare
/// `String` path, but the value expression sources `&str` via UFCS through
/// `AsRef<str>`. The bytes are copied into the view array once, identical
/// to the `String::as_str()` path.
fn vec_encoder_as_str(ctx: &LeafCtx<'_>, shape: &VecShape, base: &BaseType) -> Encoder {
    // For bare `String`, `&String` deref-coerces to `&str`; for non-String
    // bases we go through UFCS so generic-parameter and concrete-struct
    // leaves both resolve. `as_str` on a non-String/Struct/Generic base is
    // rejected at parse time, so the codegen never reaches the wildcard arm
    // — the const-fn assert in helpers.rs catches it with a clean error
    // span. We funnel the wildcard to the same `String` arm so the match
    // doesn't trip clippy's `match_same_arms` over identical bodies.
    let value_expr = match base {
        BaseType::Struct(ident, args) => {
            let ty_path = super::strategy::build_type_path(ident, args.as_ref());
            quote! { <#ty_path as ::core::convert::AsRef<str>>::as_ref(__df_derive_v) }
        }
        BaseType::Generic(ident) => {
            quote! { <#ident as ::core::convert::AsRef<str>>::as_ref(__df_derive_v) }
        }
        BaseType::String
        | BaseType::F64
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
        | BaseType::Decimal => quote! { __df_derive_v.as_str() },
    };
    let spec = VecLeafSpec::StringLike {
        value_expr,
        extra_decls: Vec::new(),
    };
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

// --- Top-level dispatcher ---

/// Returns `Some(encoder)` when the (base, transform, wrappers) triple can be
/// served by this encoder IR. After Step 4 this covers every wrapper stack
/// the parser accepts, for every primitive leaf except bare `ISize`/`USize`.
///
/// `ISize`/`USize` are kept on the legacy generic path because their
/// materialized buffer type doesn't match the field's native type — routing
/// them through `numeric_leaf` would need extra cast/clone plumbing for
/// ~zero real-world impact.
pub fn try_build_encoder(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    ctx: &LeafCtx<'_>,
) -> Option<Encoder> {
    match normalize_wrappers(wrappers) {
        WrapperKind::Leaf { option_layers: 0 } => build_leaf(base, transform, ctx),
        WrapperKind::Leaf { option_layers: 1 } => {
            let leaf = build_leaf(base, transform, ctx)?;
            Some(wrap_option(base, transform, leaf, ctx))
        }
        // Primitive multi-Option leaf shapes (`Option<Option<i32>>` etc.)
        // are not represented in the test surface — leave them on the
        // legacy primitive path. Nested multi-Option (`Option<Option<T>>`)
        // is handled in `try_build_nested_encoder`.
        WrapperKind::Leaf { option_layers: _ } => None,
        WrapperKind::Vec(shape) => {
            // `[Option, Vec, ...]` (outer-Option above a single-Vec stack)
            // over a primitive that has a typed `ListPrimitiveChunkedBuilder` /
            // `ListStringChunkedBuilder` match keeps the legacy
            // `gen_typed_list_append` fast path. The typed builder's
            // `append_iter` runs `extend_trusted_len_unchecked` over the
            // inner-Option iterator and `append_null` for outer-None, both
            // tighter than the general encoder's per-element `flat.push +
            // validity.set` plus an explicit outer validity bitmap. Bench
            // `08_complex_wrappers` measures ~10% regression when this
            // shape goes through the general path; routing back to the
            // typed builder restores baseline. The general encoder still
            // owns deeper-Vec, struct/generic, and bool/ISize/USize shapes.
            if shape.depth() == 1
                && shape.layers[0].has_outer_validity()
                && super::primitive::typed_primitive_list_info(base, transform, wrappers).is_some()
            {
                return None;
            }
            try_build_vec_encoder(base, transform, ctx, &shape)
        }
    }
}

/// Build the depth-N `vec(inner)` encoder for the `(base, transform)`
/// combinations the encoder IR covers.
///
/// Matches `build_leaf`'s coverage: bare numeric, `String`, `Bool`,
/// `Decimal` (with `DecimalToInt128`), `DateTime` (with `DateTimeToInt`),
/// `as_str` borrow, and `to_string`. Returns `None` for `ISize`/`USize`
/// (legacy generic path) and for transforms that don't apply to the base.
fn try_build_vec_encoder(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    ctx: &LeafCtx<'_>,
    shape: &VecShape,
) -> Option<Encoder> {
    match transform {
        None => try_build_vec_encoder_bare(base, ctx, shape),
        Some(PrimitiveTransform::DateTimeToInt(unit)) => match base {
            BaseType::DateTimeUtc => Some(vec_encoder_datetime(ctx, *unit, shape)),
            _ => None,
        },
        Some(PrimitiveTransform::DecimalToInt128 { precision, scale }) => match base {
            BaseType::Decimal => Some(vec_encoder_decimal(ctx, *precision, *scale, shape)),
            _ => None,
        },
        Some(PrimitiveTransform::ToString) => Some(vec_encoder_to_string(ctx, shape)),
        // `as_str` borrow path: same MBVA-based encoder as `String`, but
        // the value expression goes through UFCS (`AsRef<str>`) instead of
        // `String::as_str`. Bytes are copied into the view array once.
        Some(PrimitiveTransform::AsStr) => Some(vec_encoder_as_str(ctx, shape, base)),
    }
}

fn try_build_vec_encoder_bare(
    base: &BaseType,
    ctx: &LeafCtx<'_>,
    shape: &VecShape,
) -> Option<Encoder> {
    match base {
        BaseType::I8
        | BaseType::I16
        | BaseType::I32
        | BaseType::I64
        | BaseType::U8
        | BaseType::U16
        | BaseType::U32
        | BaseType::U64
        | BaseType::F32
        | BaseType::F64 => {
            let info = super::type_registry::numeric_info(base)?;
            // The loop binding is `&T` for Copy primitives, so dereferencing
            // produces the storage value directly. Bare and inner-Option
            // arms share the same expression because `build_vec_leaf_pieces`
            // re-binds the bare loop variable as `__df_derive_v` before
            // splicing the value expression in.
            let spec = VecLeafSpec::Numeric {
                native: info.native.clone(),
                value_expr: quote! { *__df_derive_v },
                needs_decimal_import: false,
            };
            Some(vec_encoder(ctx, &spec, shape, &info.dtype))
        }
        BaseType::String => {
            let pp = super::polars_paths::prelude();
            let leaf_dtype = quote! { #pp::DataType::String };
            let spec = VecLeafSpec::StringLike {
                value_expr: quote! { __df_derive_v.as_str() },
                extra_decls: Vec::new(),
            };
            Some(vec_encoder(ctx, &spec, shape, &leaf_dtype))
        }
        BaseType::Bool => {
            if shape.has_inner_option() {
                let pp = super::polars_paths::prelude();
                let leaf_dtype = quote! { #pp::DataType::Boolean };
                Some(vec_encoder(ctx, &VecLeafSpec::Bool, shape, &leaf_dtype))
            } else {
                Some(vec_encoder_bool_bare(ctx, shape))
            }
        }
        BaseType::ISize
        | BaseType::USize
        | BaseType::DateTimeUtc
        | BaseType::Decimal
        | BaseType::Struct(..)
        | BaseType::Generic(_) => None,
    }
}

fn vec_encoder_datetime(ctx: &LeafCtx<'_>, unit: DateTimeUnit, shape: &VecShape) -> Encoder {
    let pp = super::polars_paths::prelude();
    let unit_tokens = match unit {
        DateTimeUnit::Milliseconds => quote! { #pp::TimeUnit::Milliseconds },
        DateTimeUnit::Microseconds => quote! { #pp::TimeUnit::Microseconds },
        DateTimeUnit::Nanoseconds => quote! { #pp::TimeUnit::Nanoseconds },
    };
    let leaf_dtype = quote! {
        #pp::DataType::Datetime(#unit_tokens, ::std::option::Option::None)
    };
    let mapped_v = super::type_registry::map_primitive_expr(
        &quote! { __df_derive_v },
        Some(&PrimitiveTransform::DateTimeToInt(unit)),
        ctx.decimal128_encode_trait,
    );
    let spec = VecLeafSpec::Numeric {
        native: quote! { i64 },
        value_expr: mapped_v,
        needs_decimal_import: false,
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

fn vec_encoder_decimal(ctx: &LeafCtx<'_>, precision: u8, scale: u8, shape: &VecShape) -> Encoder {
    let pp = super::polars_paths::prelude();
    let p = precision as usize;
    let s = scale as usize;
    let leaf_dtype = quote! { #pp::DataType::Decimal(#p, #s) };
    let mapped_v = super::type_registry::map_primitive_expr(
        &quote! { __df_derive_v },
        Some(&PrimitiveTransform::DecimalToInt128 { precision, scale }),
        ctx.decimal128_encode_trait,
    );
    let spec = VecLeafSpec::Numeric {
        native: quote! { i128 },
        value_expr: mapped_v,
        needs_decimal_import: true,
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

fn vec_encoder_to_string(ctx: &LeafCtx<'_>, shape: &VecShape) -> Encoder {
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    // `to_string` materializes via `Display::fmt` into a reusable `String`
    // scratch — we splice that scratch's `as_str()` into the MBVA-push
    // expression, so the per-element work allocates the scratch once at
    // decl time and reuses on every row.
    let scratch = PopulatorIdents::primitive_str_scratch(ctx.idx);
    let value_expr = quote! {{
        use ::std::fmt::Write as _;
        #scratch.clear();
        ::std::write!(&mut #scratch, "{}", __df_derive_v).unwrap();
        #scratch.as_str()
    }};
    let spec = VecLeafSpec::StringLike {
        value_expr,
        extra_decls: vec![
            quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); },
        ],
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

fn build_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    ctx: &LeafCtx<'_>,
) -> Option<Encoder> {
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
            | BaseType::F64 => Some(numeric_leaf(ctx, base)),
            BaseType::String => Some(string_leaf(ctx)),
            BaseType::Bool => Some(bool_leaf(ctx)),
            BaseType::ISize
            | BaseType::USize
            | BaseType::DateTimeUtc
            | BaseType::Decimal
            | BaseType::Struct(..)
            | BaseType::Generic(_) => None,
        },
        Some(PrimitiveTransform::DateTimeToInt(unit)) => match base {
            BaseType::DateTimeUtc => Some(datetime_leaf(ctx, *unit)),
            _ => None,
        },
        Some(PrimitiveTransform::DecimalToInt128 { precision, scale }) => match base {
            BaseType::Decimal => Some(decimal_leaf(ctx, *precision, *scale)),
            _ => None,
        },
        Some(PrimitiveTransform::ToString) => Some(as_string_leaf(ctx)),
        Some(PrimitiveTransform::AsStr) => {
            let ty_path = match base {
                BaseType::String => quote! { ::std::string::String },
                BaseType::Struct(ident, args) => {
                    super::strategy::build_type_path(ident, args.as_ref())
                }
                BaseType::Generic(ident) => quote! { #ident },
                _ => return None,
            };
            Some(as_str_leaf(ctx, &ty_path, base))
        }
    }
}

fn wrap_option(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    inner: Encoder,
    ctx: &LeafCtx<'_>,
) -> Encoder {
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
            | BaseType::F64 => option_for_numeric(ctx, base, inner),
            BaseType::String => option_for_string_like(ctx, vec![], inner),
            BaseType::Bool => option_for_bool(ctx, inner),
            BaseType::ISize
            | BaseType::USize
            | BaseType::DateTimeUtc
            | BaseType::Decimal
            | BaseType::Struct(..)
            | BaseType::Generic(_) => {
                unreachable!("wrap_option reached for a leaf shape build_leaf returns None for")
            }
        },
        Some(PrimitiveTransform::DateTimeToInt(unit)) => option_for_datetime(ctx, *unit, inner),
        Some(PrimitiveTransform::DecimalToInt128 { precision, scale }) => {
            option_for_decimal(ctx, *precision, *scale, inner)
        }
        Some(PrimitiveTransform::ToString) => {
            // `as_string` has a `String` scratch on top of the MBVA pair —
            // pass it as an extra decl into the shared option-string-like
            // combinator so the layout exactly matches the prior emission.
            let scratch = PopulatorIdents::primitive_str_scratch(ctx.idx);
            let extra = vec![
                quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); },
            ];
            option_for_string_like(ctx, extra, inner)
        }
        Some(PrimitiveTransform::AsStr) => option_for_as_str(ctx, inner),
    }
}

// --- Nested-struct/generic encoder paths (CollectThenBulk leaves) ---
//
// Step 3 ports the seven nested-struct/generic shapes (`[]`, `[Option]`,
// `[Vec]`, `[Option, Vec]`, `[Vec, Option]`, `[Option, Vec, Option]`,
// `[Vec, Vec]`) into the encoder IR. Each shape is built up from a single
// `CollectThenBulk` leaf (which knows how to call
// `<T as Columnar>::columnar_from_refs(&refs)`) plus the wrapper-stack-shaped
// gather/scatter machinery in this section.
//
// The invariant: every `LargeListArray::new` routes through the in-scope free
// helper `__df_derive_assemble_list_series_unchecked` (defined at the top of
// each derive's `const _: () = { ... };` scope), keeping `unsafe` out of any
// `Self`-bearing impl method so `clippy::unsafe_derive_deserialize` stays
// silent on downstream `#[derive(ToDataFrame, Deserialize)]` types.
//
// Every shape produces an `EncoderFinish::Multi { columnar }` because the
// inner `DataFrame` carries one column per inner schema entry of `T`. The
// block pushes one Series per inner schema column onto the call site's
// `columns` vec, with the parent name prefixed onto each inner column name.

/// Per-call-site context for nested-struct/generic encoders. Carries the
/// `polars-arrow` crate root (so the combinators don't re-resolve it per
/// call) plus the type-as-path expression and the fully-qualified trait
/// paths used in UFCS calls (`<#ty as #columnar_trait>::columnar_from_refs`,
/// `<#ty as #to_df_trait>::schema`).
pub struct NestedLeafCtx<'a> {
    pub access: &'a TokenStream,
    pub idx: usize,
    pub parent_name: &'a str,
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a TokenStream,
    pub to_df_trait: &'a TokenStream,
    pub pa_root: &'a TokenStream,
}

/// Per-shape identifier bundle for the nested encoder paths. Computing these
/// once at the top of each shape builder keeps the per-shape body focused on
/// the gather/scatter logic.
struct NestedIdents {
    /// `Vec<&T>` flat ref accumulator.
    flat: syn::Ident,
    /// `Vec<Option<IdxSize>>` per-element positions for the inner-Option
    /// scatter case.
    positions: syn::Ident,
    /// Inner `DataFrame` returned by `columnar_from_refs`.
    df: syn::Ident,
    /// `IdxCa` built from `positions` for the scatter case.
    take: syn::Ident,
    /// Total inner-element count (used by precount + outer-list capacity).
    total: syn::Ident,
}

impl NestedIdents {
    fn new(idx: usize) -> Self {
        Self {
            flat: format_ident!("__df_derive_gen_flat_{}", idx),
            positions: format_ident!("__df_derive_gen_pos_{}", idx),
            df: format_ident!("__df_derive_gen_df_{}", idx),
            take: format_ident!("__df_derive_gen_take_{}", idx),
            total: format_ident!("__df_derive_gen_total_{}", idx),
        }
    }
}

/// Build the per-column emit body that iterates `<T as ToDataFrame>::schema()?`
/// and pushes each inner-Series-yielding expression onto `columns` with the
/// parent name prefixed. The schema name is exposed as `__df_derive_col_name:
/// &str` and the dtype as `__df_derive_dtype: &polars::DataType` so per-column
/// expressions can reference both.
fn nested_consume_columns(
    parent_name: &str,
    to_df_trait: &TokenStream,
    ty: &TokenStream,
    series_expr: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    quote! {
        for (__df_derive_col_name, __df_derive_dtype) in
            <#ty as #to_df_trait>::schema()?
        {
            let __df_derive_col_name: &str = __df_derive_col_name.as_str();
            let __df_derive_dtype: &#pp::DataType = &__df_derive_dtype;
            {
                let __df_derive_prefixed = ::std::format!(
                    "{}.{}", #parent_name, __df_derive_col_name,
                );
                let __df_derive_inner: #pp::Series = #series_expr;
                let __df_derive_named = __df_derive_inner
                    .with_name(__df_derive_prefixed.as_str().into());
                columns.push(__df_derive_named.into());
            }
        }
    }
}

/// Build the bare-leaf nested encoder (`payload: T`). Gathers refs into
/// `Vec<&T>`, calls `columnar_from_refs` once, and per inner schema column
/// pulls the materialized `Series` straight out of the resulting `DataFrame`
/// (no list-array wrapping; the parent column is the inner column).
fn nested_leaf_encoder(ctx: &NestedLeafCtx<'_>) -> Encoder {
    let NestedLeafCtx {
        access,
        idx,
        parent_name,
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let df = &ids.df;
    let inner_expr = quote! {
        #df.column(__df_derive_col_name)?
            .as_materialized_series()
            .clone()
    };
    let columnar = nested_consume_columns(parent_name, to_df_trait, ty, &inner_expr);
    let setup = quote! {
        let #flat: ::std::vec::Vec<&#ty> = items
            .iter()
            .map(|__df_derive_it| &(#access))
            .collect();
        let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
    };
    let columnar_block = quote! {{ #setup #columnar }};
    Encoder {
        decls: Vec::new(),
        push: TokenStream::new(),
        option_push: None,
        finish: EncoderFinish::Multi {
            columnar: columnar_block,
        },
        kind: LeafKind::CollectThenBulk,
        offset_depth: 0,
    }
}

/// `option(nested_leaf)` — `[Option]` (or any consecutive run of `Option`s
/// over a struct/generic, since Polars folds nested Nones into one validity
/// bit). For `option_layers >= 2`, the caller pre-collapses the access into
/// an `Option<&T>` value-expression; the scan then reads the value directly
/// without a `&`.
///
/// Splits each row's `Option<T>` into a flat ref slice plus a
/// `Vec<Option<IdxSize>>` of positions. Three runtime branches:
/// - all None: emit one typed-null Series of length `items.len()` per inner
///   schema column.
/// - all Some (no scatter needed): pull each column straight from the inner
///   `DataFrame`, no `take`.
/// - mixed: build an `IdxCa` over positions and `take` per inner column to
///   scatter values back over the original row positions.
fn nested_option_encoder_collapsed(ctx: &NestedLeafCtx<'_>, option_layers: usize) -> Encoder {
    // For `option_layers >= 2`, `#access` is an `as_ref().and_then(...)`
    // chain returning `Option<&T>` directly — we match it by value.
    // For `option_layers == 1`, `#access` is the raw `&Option<T>` field
    // expression — we match by reference. The two arms produce slightly
    // different scans because the bound `__df_derive_v` is `&T` either way,
    // but the surrounding match expression differs.
    let access_ts = ctx.access.clone();
    let match_expr = if option_layers >= 2 {
        quote! { (#access_ts) }
    } else {
        quote! { &(#access_ts) }
    };
    nested_option_encoder_impl(ctx, &match_expr)
}

fn nested_option_encoder_impl(ctx: &NestedLeafCtx<'_>, match_expr: &TokenStream) -> Encoder {
    let NestedLeafCtx {
        access: _,
        idx,
        parent_name,
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let pp = super::polars_paths::prelude();
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let positions = &ids.positions;
    let df = &ids.df;
    let take = &ids.take;

    let direct_inner = quote! {
        #df.column(__df_derive_col_name)?
            .as_materialized_series()
            .clone()
    };
    let take_inner = quote! {{
        let __df_derive_inner_full = #df
            .column(__df_derive_col_name)?
            .as_materialized_series();
        __df_derive_inner_full.take(&#take)?
    }};
    let null_inner = quote! {
        #pp::Series::new_empty("".into(), __df_derive_dtype)
            .extend_constant(#pp::AnyValue::Null, items.len())?
    };

    let scan = quote! {
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::new();
        let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for __df_derive_it in items {
            match #match_expr {
                ::std::option::Option::Some(__df_derive_v) => {
                    #positions.push(::std::option::Option::Some(
                        #flat.len() as #pp::IdxSize,
                    ));
                    #flat.push(__df_derive_v);
                }
                ::std::option::Option::None => {
                    #positions.push(::std::option::Option::None);
                }
            }
        }
    };
    let consume_direct = nested_consume_columns(parent_name, to_df_trait, ty, &direct_inner);
    let consume_take = nested_consume_columns(parent_name, to_df_trait, ty, &take_inner);
    let consume_null = nested_consume_columns(parent_name, to_df_trait, ty, &null_inner);
    let columnar_block = quote! {{
        #scan
        if #flat.is_empty() {
            #consume_null
        } else if #flat.len() == items.len() {
            let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
            #consume_direct
        } else {
            let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
            let #take: #pp::IdxCa =
                <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #positions.iter().copied(),
                );
            #consume_take
        }
    }};
    Encoder {
        decls: Vec::new(),
        push: TokenStream::new(),
        option_push: None,
        finish: EncoderFinish::Multi {
            columnar: columnar_block,
        },
        kind: LeafKind::CollectThenBulk,
        offset_depth: 0,
    }
}

// --- Generalized depth-N nested encoder ---

/// Per-layer ident set for the depth-N nested encoder. Layer 0 is the
/// outermost `Vec`. `offsets` accumulates offset entries (one per child-list
/// inside this layer); `offsets_buf` is the frozen `OffsetsBuffer`. `validity`
/// holds the layer-level `MutableBitmap` when this layer's outer Option is
/// present. `bind` is the per-layer iteration binding (mirrors the primitive
/// `VecLayerIdents` pattern).
struct NestedLayerIdents {
    offsets: syn::Ident,
    offsets_buf: syn::Ident,
    validity_mb: syn::Ident,
    validity_bm: syn::Ident,
    bind: syn::Ident,
}

fn nested_layer_idents(idx: usize, depth: usize) -> Vec<NestedLayerIdents> {
    (0..depth)
        .map(|i| NestedLayerIdents {
            offsets: format_ident!("__df_derive_n_off_{}_{}", idx, i),
            offsets_buf: format_ident!("__df_derive_n_off_buf_{}_{}", idx, i),
            validity_mb: format_ident!("__df_derive_n_valmb_{}_{}", idx, i),
            validity_bm: format_ident!("__df_derive_n_valbm_{}_{}", idx, i),
            bind: format_ident!("__df_derive_n_bind_{}_{}", idx, i),
        })
        .collect()
}

/// Build the depth-N nested vec encoder for an arbitrary [`VecShape`].
/// Handles per-layer outer-Option (validity bitmap), inner-Option
/// (per-element positions + scatter via `IdxCa::take`), and any mix
/// thereof. Replaces the seven hand-written shape variants.
#[allow(clippy::too_many_lines)]
fn nested_vec_encoder_general(ctx: &NestedLeafCtx<'_>, shape: &VecShape) -> Encoder {
    let NestedLeafCtx {
        access,
        idx,
        parent_name,
        ty,
        columnar_trait,
        to_df_trait,
        pa_root,
    } = *ctx;
    let pp = super::polars_paths::prelude();
    let depth = shape.depth();
    let layers = nested_layer_idents(idx, depth);
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let positions = &ids.positions;
    let df = &ids.df;
    let take = &ids.take;
    let total = &ids.total;

    let scan_body = build_nested_scan_body(access, shape, &layers, flat, positions, total, ty);
    let validity_freeze = build_nested_validity_freeze(shape, &layers, pa_root);
    let offsets_freeze = build_nested_offsets_freeze(&layers, pa_root);

    // Per-column wrap expressions. We need three branches:
    // - filled-direct: `df` exists, positions match flat 1:1 (or no inner
    //   option) → wrap each `df.column(name)` chunk in N list layers.
    // - filled-take: `df` exists, positions has Nones → take with IdxCa,
    //   wrap result in N list layers.
    // - all-empty (flat empty): wrap an empty Series in N list layers.

    let inner_col_direct = quote! {
        #df.column(__df_derive_col_name)?
            .as_materialized_series()
            .clone()
    };
    let inner_col_take = quote! {{
        let __df_derive_inner_full = #df
            .column(__df_derive_col_name)?
            .as_materialized_series();
        __df_derive_inner_full.take(&#take)?
    }};
    let inner_col_empty = quote! {
        #pp::Series::new_empty("".into(), __df_derive_dtype)
    };
    // All-absent: every element slot is `None`, but the outer offsets are
    // non-zero (each outer row carries inner-Vec lengths > 0). The inner
    // chunk must be a typed-null Series of length `total` so the offsets
    // buffer's max value (which equals `total`) doesn't exceed the chunk's
    // length. Without inner-Option, this branch is unreachable — zero
    // total leaves implies zero outer-list members.
    let inner_col_all_absent = quote! {
        #pp::Series::new_empty("".into(), __df_derive_dtype)
            .extend_constant(#pp::AnyValue::Null, #total)?
    };

    let series_direct = build_nested_layer_wrap(&layers, shape, &inner_col_direct, &pp);
    let series_take = build_nested_layer_wrap(&layers, shape, &inner_col_take, &pp);
    let series_empty = build_nested_layer_wrap(&layers, shape, &inner_col_empty, &pp);
    let series_all_absent = build_nested_layer_wrap(&layers, shape, &inner_col_all_absent, &pp);

    let consume_direct = nested_consume_columns(parent_name, to_df_trait, ty, &series_direct);
    let consume_take = nested_consume_columns(parent_name, to_df_trait, ty, &series_take);
    let consume_empty = nested_consume_columns(parent_name, to_df_trait, ty, &series_empty);
    let consume_all_absent =
        nested_consume_columns(parent_name, to_df_trait, ty, &series_all_absent);
    let columnar_block = if shape.has_inner_option() {
        // 4-branch dispatch for the inner-Option case:
        // - total == 0: no leaf slots at all (every outer Vec was empty
        //   or every outer Option was None) → empty inner Series.
        // - flat.is_empty() (but total > 0): every leaf slot was None →
        //   typed-null Series of length total, offsets reference it.
        // - flat.len() == total: every leaf slot was Some → direct.
        // - else: mixed → take.
        quote! {{
            #scan_body
            #validity_freeze
            #offsets_freeze
            if #total == 0 {
                #consume_empty
            } else if #flat.is_empty() {
                #consume_all_absent
            } else if #flat.len() == #total {
                let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
                #consume_direct
            } else {
                let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
                let #take: #pp::IdxCa =
                    <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                        "".into(),
                        #positions.iter().copied(),
                    );
                #consume_take
            }
        }}
    } else {
        // No inner-Option: total == flat.len(). Two branches: empty
        // when flat is empty (no leaves), direct otherwise.
        quote! {{
            #scan_body
            #validity_freeze
            #offsets_freeze
            if #flat.is_empty() {
                #consume_empty
            } else {
                let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
                #consume_direct
            }
        }}
    };
    Encoder {
        decls: Vec::new(),
        push: TokenStream::new(),
        option_push: None,
        finish: EncoderFinish::Multi {
            columnar: columnar_block,
        },
        kind: LeafKind::CollectThenBulk,
        offset_depth: depth,
    }
}

/// Scan the input building flat refs, per-layer offsets vecs, per-layer
/// validity bitmaps, and (when `has_inner_option`) per-element positions.
#[allow(clippy::items_after_statements, clippy::too_many_lines)]
fn build_nested_scan_body(
    access: &TokenStream,
    shape: &VecShape,
    layers: &[NestedLayerIdents],
    flat: &syn::Ident,
    positions: &syn::Ident,
    total: &syn::Ident,
    ty: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let pa_root = super::polars_paths::polars_arrow_root();
    let depth = shape.depth();

    // Per-layer counter for sizing offsets vecs and validity bitmaps.
    // `total_layer_{i}` counts the number of child-lists inside layer `i`
    // (i.e. how many entries layer `i+1` will accumulate). Layer 0 is sized
    // by `items.len()` directly. `total_leaves` (== `total`) is used for the
    // flat ref vec capacity (and the positions vec under has_inner_option).
    let layer_counters: Vec<syn::Ident> = (0..depth.saturating_sub(1))
        .map(|i| format_ident!("__df_derive_n_total_layer_{}", i))
        .collect();

    // Build the inner iter body for layer `cur`. `vec_bind` is the expression
    // that holds the inner Vec (after Option-unwrap). At the deepest layer
    // we iterate elements (and optionally scatter into positions).
    //
    // The scan loop must NOT increment the per-layer counters that the
    // precount loop sized — those counters are dead-store after the precount
    // and re-incrementing them during scan only inflates loop bodies (LLVM
    // does eliminate the dead store, but the path is sensitive to exactly
    // how the loop is shaped). Counters live exclusively in `build_pre_iter`.
    fn build_iter(
        shape: &VecShape,
        layers: &[NestedLayerIdents],
        flat: &syn::Ident,
        positions: &syn::Ident,
        cur: usize,
        vec_bind: &TokenStream,
    ) -> TokenStream {
        let depth = shape.depth();
        if cur + 1 == depth {
            // Innermost: iterate values. Inner-Option branches per element.
            if shape.has_inner_option() {
                let pp = super::polars_paths::prelude();
                quote! {
                    for __df_derive_maybe in #vec_bind.iter() {
                        match __df_derive_maybe {
                            ::std::option::Option::Some(__df_derive_v) => {
                                #positions.push(::std::option::Option::Some(
                                    #flat.len() as #pp::IdxSize,
                                ));
                                #flat.push(__df_derive_v);
                            }
                            ::std::option::Option::None => {
                                #positions.push(::std::option::Option::None);
                            }
                        }
                    }
                }
            } else {
                quote! {
                    for __df_derive_v in #vec_bind.iter() {
                        #flat.push(__df_derive_v);
                    }
                }
            }
        } else {
            let inner_bind = &layers[cur + 1].bind;
            let inner_layer_body = build_layer(
                shape,
                layers,
                flat,
                positions,
                cur + 1,
                &quote! { #inner_bind },
            );
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_layer_body
                }
            }
        }
    }

    fn build_layer(
        shape: &VecShape,
        layers: &[NestedLayerIdents],
        flat: &syn::Ident,
        positions: &syn::Ident,
        cur: usize,
        bind: &TokenStream,
    ) -> TokenStream {
        let depth = shape.depth();
        let layer = &layers[cur];
        let offsets = &layer.offsets;
        // `offsets.push(...)` value: for the innermost layer it's flat.len()
        // (or positions.len() when inner-option scatters); for outer layers
        // it's the next-inner-layer's offsets length minus 1.
        let offsets_post = if cur + 1 == depth {
            if shape.has_inner_option() {
                quote! { #positions.len() }
            } else {
                quote! { #flat.len() }
            }
        } else {
            let inner_offsets = &layers[cur + 1].offsets;
            quote! { (#inner_offsets.len() - 1) }
        };
        let inner_iter = if shape.layers[cur].has_outer_validity() {
            let validity = &layer.validity_mb;
            let inner_vec_bind = format_ident!("__df_derive_n_some_{}", cur);
            let inner_iter = build_iter(
                shape,
                layers,
                flat,
                positions,
                cur,
                &quote! { #inner_vec_bind },
            );
            quote! {
                match #bind {
                    ::std::option::Option::Some(#inner_vec_bind) => {
                        #validity.push(true);
                        #inner_iter
                    }
                    ::std::option::Option::None => {
                        #validity.push(false);
                    }
                }
            }
        } else {
            build_iter(shape, layers, flat, positions, cur, bind)
        };
        quote! {
            #inner_iter
            #offsets.push(#offsets_post as i64);
        }
    }

    // Precount: walk the same structure and tally totals.
    fn build_pre_iter(
        shape: &VecShape,
        layers: &[NestedLayerIdents],
        layer_counters: &[syn::Ident],
        total: &syn::Ident,
        cur: usize,
        vec_bind: &TokenStream,
    ) -> TokenStream {
        let depth = shape.depth();
        if cur + 1 == depth {
            quote! { #total += #vec_bind.len(); }
        } else {
            let inner_bind = &layers[cur + 1].bind;
            let counter = &layer_counters[cur];
            let inner_pre = build_pre_layer(
                shape,
                layers,
                layer_counters,
                total,
                cur + 1,
                &quote! { #inner_bind },
            );
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_pre
                    #counter += 1;
                }
            }
        }
    }

    fn build_pre_layer(
        shape: &VecShape,
        layers: &[NestedLayerIdents],
        layer_counters: &[syn::Ident],
        total: &syn::Ident,
        cur: usize,
        bind: &TokenStream,
    ) -> TokenStream {
        if shape.layers[cur].has_outer_validity() {
            let inner_vec_bind = format_ident!("__df_derive_n_pre_some_{}", cur);
            let inner = build_pre_iter(
                shape,
                layers,
                layer_counters,
                total,
                cur,
                &quote! { #inner_vec_bind },
            );
            quote! {
                if let ::std::option::Option::Some(#inner_vec_bind) = #bind {
                    #inner
                }
            }
        } else {
            build_pre_iter(shape, layers, layer_counters, total, cur, bind)
        }
    }

    let layer0_iter_src = quote! { (&(#access)) };
    let scan_iter_body = build_layer(shape, layers, flat, positions, 0, &layer0_iter_src);
    let pre_iter_body = build_pre_layer(shape, layers, &layer_counters, total, 0, &layer0_iter_src);

    // Allocate offsets vecs with capacity from the layer counters. Layer 0
    // is `items.len() + 1`; deeper layers use the layer counter.
    let mut offsets_decls: Vec<TokenStream> = Vec::with_capacity(depth);
    for (i, layer) in layers.iter().enumerate() {
        let offsets = &layer.offsets;
        let cap = if i == 0 {
            quote! { items.len() + 1 }
        } else {
            let counter = &layer_counters[i - 1];
            quote! { #counter + 1 }
        };
        offsets_decls.push(quote! {
            let mut #offsets: ::std::vec::Vec<i64> = ::std::vec::Vec::with_capacity(#cap);
            #offsets.push(0);
        });
    }
    let mut validity_decls: Vec<TokenStream> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if !shape.layers[i].has_outer_validity() {
            continue;
        }
        let validity = &layer.validity_mb;
        let cap = if i == 0 {
            quote! { items.len() }
        } else {
            let counter = &layer_counters[i - 1];
            quote! { #counter }
        };
        validity_decls.push(quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(#cap);
        });
    }
    let counter_decls = layer_counters
        .iter()
        .map(|c| quote! { let mut #c: usize = 0; });
    let positions_decl = if shape.has_inner_option() {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total);
        }
    } else {
        TokenStream::new()
    };
    quote! {
        let mut #total: usize = 0;
        #(#counter_decls)*
        for __df_derive_it in items {
            #pre_iter_body
        }
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::with_capacity(#total);
        #positions_decl
        #(#offsets_decls)*
        #(#validity_decls)*
        for __df_derive_it in items {
            #scan_iter_body
        }
    }
}

/// Freeze the per-layer `MutableBitmap` validity buffers into immutable
/// `Bitmap`s (only for layers whose Option is present). Each layer's bitmap
/// is named `validity_bm_<idx>_<layer>` post-freeze.
fn build_nested_validity_freeze(
    shape: &VecShape,
    layers: &[NestedLayerIdents],
    pa_root: &TokenStream,
) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if !shape.layers[i].has_outer_validity() {
            continue;
        }
        let mb = &layer.validity_mb;
        let bm = &layer.validity_bm;
        out.push(quote! {
            let #bm: #pa_root::bitmap::Bitmap =
                <#pa_root::bitmap::Bitmap as ::core::convert::From<
                    #pa_root::bitmap::MutableBitmap,
                >>::from(#mb);
        });
    }
    quote! { #(#out)* }
}

/// Freeze each layer's offsets vec into an `OffsetsBuffer`. Layer 0 is
/// always populated. When all layers are empty (flat is empty), the
/// offsets vecs still contain at least the leading 0; the freeze still
/// succeeds because `OffsetsBuffer::try_from(vec![0])` is valid.
///
/// The freeze consumes each offsets `Vec<i64>` into the buffer (no clone).
/// `OffsetsBuffer` is `Arc`-backed, so subsequent uses (one per branch in
/// the four-way dispatch, plus per-layer wraps) clone cheaply by bumping
/// the refcount.
fn build_nested_offsets_freeze(layers: &[NestedLayerIdents], pa_root: &TokenStream) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::new();
    for layer in layers {
        let offsets = &layer.offsets;
        let buf = &layer.offsets_buf;
        out.push(quote! {
            let #buf: #pa_root::offset::OffsetsBuffer<i64> =
                #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
        });
    }
    quote! { #(#out)* }
}

/// Wrap the supplied inner-column expression in `depth` `LargeListArray::new`
/// layers (innermost-first, outermost-last) and route the outermost through
/// `__df_derive_assemble_list_series_unchecked`. Per-layer validity bitmaps
/// (when their Option is present) ride under each `LargeListArray`.
fn build_nested_layer_wrap(
    layers: &[NestedLayerIdents],
    shape: &VecShape,
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let depth = layers.len();
    let mut block: Vec<TokenStream> = Vec::new();
    block.push(quote! {
        let __df_derive_inner_col: #pp::Series = #inner_col_expr;
        let __df_derive_inner_rech = __df_derive_inner_col.rechunk();
        let __df_derive_inner_chunk: #pp::ArrayRef =
            __df_derive_inner_rech.chunks()[0].clone();
    });
    let mut prev_arr = format_ident!("__df_derive_inner_chunk");
    for cur in (0..depth).rev() {
        let layer = &layers[cur];
        let buf = &layer.offsets_buf;
        let arr_id = format_ident!("__df_derive_n_arr_{}", cur);
        let validity_expr = if shape.layers[cur].has_outer_validity() {
            let bm = &layer.validity_bm;
            quote! { ::std::option::Option::Some(::std::clone::Clone::clone(&#bm)) }
        } else {
            quote! { ::std::option::Option::None }
        };
        let prev = prev_arr.clone();
        // The first wrap consumes the inner chunk (an `ArrayRef`); subsequent
        // wraps consume the previous LargeListArray boxed as ArrayRef.
        let prev_payload = if cur == depth - 1 {
            quote! { #prev }
        } else {
            quote! { ::std::boxed::Box::new(#prev) as #pp::ArrayRef }
        };
        let pa_root = super::polars_paths::polars_arrow_root();
        // Read the chunk's arrow dtype: `ArrayRef`'s dtype method (the
        // first wrap) and `LargeListArray::dtype()` (subsequent wraps)
        // both proxy through the `Array` trait.
        let dtype_src = if cur == depth - 1 {
            quote! { #prev.dtype().clone() }
        } else {
            quote! { #pa_root::array::Array::dtype(&#prev).clone() }
        };
        block.push(quote! {
            let #arr_id: #pp::LargeListArray = #pp::LargeListArray::new(
                #pp::LargeListArray::default_datatype(#dtype_src),
                ::std::clone::Clone::clone(&#buf),
                #prev_payload,
                #validity_expr,
            );
        });
        prev_arr = arr_id;
    }
    // Wrap the per-leaf logical dtype in `(depth - 1)` extra `List<>` layers
    // to construct what `__df_derive_assemble_list_series_unchecked` expects
    // (the helper wraps once more, yielding the full N-layer List nesting).
    let mut helper_logical = quote! { (*__df_derive_dtype).clone() };
    for _ in 0..depth.saturating_sub(1) {
        helper_logical = quote! { #pp::DataType::List(::std::boxed::Box::new(#helper_logical)) };
    }
    let outer = prev_arr;
    quote! {{
        #(#block)*
        __df_derive_assemble_list_series_unchecked(
            #outer,
            #helper_logical,
        )
    }}
}

/// Top-level dispatcher for the nested-struct/generic encoder paths.
/// After Step 4 this covers every wrapper stack the parser accepts —
/// the `[]` and `[Option]` shapes use dedicated leaf encoders; every
/// `Vec`-bearing shape (including deep nestings, mid-stack `Option`s,
/// outer-list validity) routes through the depth-N general encoder.
pub fn build_nested_encoder(wrappers: &[Wrapper], ctx: &NestedLeafCtx<'_>) -> Encoder {
    match normalize_wrappers(wrappers) {
        WrapperKind::Leaf { option_layers: 0 } => nested_leaf_encoder(ctx),
        WrapperKind::Leaf {
            option_layers: layers,
        } => {
            // Collapse N consecutive Options into a single `Option<&T>`
            // before invoking the option-leaf encoder. Polars folds every
            // nested None into one validity bit, so `Some(None)` and
            // outer `None` produce the same `AnyValue::Null`. The
            // intermediate access expression is `(...).as_ref().and_then(...)`
            // which evaluates to `Option<&T>` and matches the option-leaf
            // encoder's expected access type for the single-Option case.
            let collapsed_access = if layers >= 2 {
                let chain = collapse_options_to_ref(ctx.access, layers);
                quote! { (#chain) }
            } else {
                ctx.access.clone()
            };
            let new_ctx = NestedLeafCtx {
                access: &collapsed_access,
                idx: ctx.idx,
                parent_name: ctx.parent_name,
                ty: ctx.ty,
                columnar_trait: ctx.columnar_trait,
                to_df_trait: ctx.to_df_trait,
                pa_root: ctx.pa_root,
            };
            nested_option_encoder_collapsed(&new_ctx, layers)
        }
        WrapperKind::Vec(shape) => nested_vec_encoder_general(ctx, &shape),
    }
}
