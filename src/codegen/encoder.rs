//! Encoder IR: a compositional encoder model for per-field `DataFrame` columnization.
//!
//! Each leaf encoder knows how to emit (decls, push, finish) for one base type.
//! The `option(inner)` combinator wraps a leaf to add `Option<...>` semantics.
//! Per-field codegen folds the wrapper stack right-to-left over the leaf to
//! assemble the final emission. Step 1 covers leaves and `option(inner)`;
//! `vec(inner)` is added in Step 2.
//!
//! Each leaf carries two push token streams: `bare_push` for the unwrapped
//! shape, and `option_push` for the `[Option]` shape. The split lets the
//! `bool` leaf override the option case with a 3-arm match (so `Some(false)`
//! is a true no-op against a values bitmap pre-filled with `false`).

use crate::ir::{BaseType, DateTimeUnit, PrimitiveTransform, Wrapper};
use proc_macro2::TokenStream;
use quote::quote;

use super::populator_idents::PopulatorIdents;

/// How a leaf consumes values. Step 1 only emits `PerElementPush` leaves.
/// `CollectThenBulk` is reserved for nested struct leaves added in Step 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    /// Leaf consumes one value at a time via a `push` token stream.
    PerElementPush,
    /// Leaf collects refs across all rows then performs one bulk encode call.
    #[allow(dead_code)]
    CollectThenBulk,
}

/// Per-field encoder state. `decls` and `finish_series` are emitted once at the
/// top/bottom of the columnar populator pipeline; `push` is spliced inside the
/// per-row loop.
///
/// `finish_series` is a token expression that evaluates to a
/// `polars::prelude::Series`. Outer call sites — the columnar pipeline (which
/// pushes columns onto `columns`) and the vec-anyvalues pipeline (which wraps
/// each Series in `AnyValue::List(...)`) — apply their own wrap to the same
/// expression. Keeping the wrap out of the encoder lets one builder serve both
/// contexts without duplicating per-shape finish logic.
pub struct Encoder {
    pub decls: Vec<TokenStream>,
    /// Push tokens used when this encoder is the top of the wrapper stack.
    pub push: TokenStream,
    /// Push tokens specifically for an outer `option(...)` wrapper. `None`
    /// makes the option combinator generate a generic 2-arm match.
    pub option_push: Option<TokenStream>,
    pub finish_series: TokenStream,
    pub kind: LeafKind,
    /// 0 for leaves, +1 per `vec` layer. Used by Step 2.
    #[allow(dead_code)]
    pub offset_depth: usize,
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
        finish_series,
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
        finish_series,
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
        finish_series,
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
        finish_series: string_chunked_series(name, &arr_expr),
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
        finish_series,
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
        finish_series,
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
        finish_series,
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
        finish_series,
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
        finish_series,
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// Option combinator for `DateTime`: switches to `Vec<Option<i64>>`. The
/// finish path is identical structurally (`Series::new` + cast); only the
/// element type changes — so we can reuse the inner `finish_series`.
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
        finish_series,
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
        finish_series,
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
        finish_series,
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
        finish_series,
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

// --- Top-level dispatcher ---

/// Returns `Some(encoder)` when the (base, transform, wrappers) triple can be
/// served by this encoder IR — currently the `[]` and `[Option]` shapes for
/// every primitive leaf except bare `ISize`/`USize`. Returns `None` for
/// anything with a `Vec` wrapper (Step 2) or for nested struct leaves
/// (Step 3) and for ISize/USize bare/option (kept on the legacy generic
/// path because their materialized buffer type doesn't match the field's
/// native type — routing them through `numeric_leaf` would need extra
/// cast/clone plumbing for ~zero real-world impact; deferred).
pub fn try_build_encoder(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    ctx: &LeafCtx<'_>,
) -> Option<Encoder> {
    let opt = match wrappers {
        [] => false,
        [Wrapper::Option] => true,
        _ => return None,
    };
    let leaf = build_leaf(base, transform, ctx)?;
    if !opt {
        return Some(leaf);
    }
    Some(wrap_option(base, transform, leaf, ctx))
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
