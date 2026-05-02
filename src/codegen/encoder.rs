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
use quote::{format_ident, quote};

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

/// Build the entire `vec(inner)` (and `vec(vec(inner))`) emit block.
///
/// `depth` is the number of outer `Vec` layers (1 or 2). `has_inner_option`
/// tracks whether the deepest layer is `Option<T>` — when true, the leaf
/// gets a parallel pre-filled validity bitmap and the per-leaf push has
/// `Some(v)`/`None` arms. The bulk-fusion contract: both depths share one
/// flat leaf buffer and one leaf-array build at the deepest layer; depth-2
/// just stacks an extra `for w in v.iter()` loop and an extra outer-offsets
/// vec.
///
/// Returns the single `let __df_derive_field_series_<idx> = { ... };`
/// declaration the caller splices into the populator's pre-loop decls.
fn vec_emit_decl(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    depth: usize,
    has_inner_option: bool,
    leaf_dtype_tokens: &TokenStream,
) -> TokenStream {
    let pa_root = super::polars_paths::polars_arrow_root();
    let pp = super::polars_paths::prelude();
    let access = ctx.access;
    let series_local = vec_encoder_series_local(ctx.idx);

    let leaf_bind = format_ident!("__df_derive_v");
    let outer_bind = format_ident!("__df_derive_inner");

    let (precount_decls, leaf_capacity_expr, inner_offsets_cap) =
        vec_precount_pieces(access, depth, &outer_bind);

    let (leaf_storage_decls, per_elem_push, leaf_arr_expr) =
        build_vec_leaf_pieces(spec, has_inner_option, &leaf_capacity_expr, &pa_root);

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
    let push_loops = vec_push_loops(
        access,
        depth,
        &outer_bind,
        &leaf_bind,
        &per_elem_push,
        &leaf_offsets_post_push,
    );
    let offsets_decls = vec_offsets_decls(depth, inner_offsets_cap.as_ref());
    let final_assemble = vec_final_assemble(depth, leaf_dtype_tokens, &pa_root, &pp);

    quote! {
        let #series_local: #pp::Series = {
            #extra_imports
            #precount_decls
            #leaf_storage_decls
            #offsets_decls
            #push_loops
            let __df_derive_leaf_arr = { #leaf_arr_expr };
            #final_assemble
        };
    }
}

/// Precount loop + capacity-expression bookkeeping for the bulk-vec emit.
///
/// At depth 1 the leaf count is one running total (sum of inner lengths).
/// At depth 2 we additionally count the mid-level lists so the inner-offsets
/// vec is sized correctly. Returns `(decls, leaf_capacity, inner_offsets_cap)`
/// — the third element is `Some` only at depth 2.
fn vec_precount_pieces(
    access: &TokenStream,
    depth: usize,
    outer_bind: &syn::Ident,
) -> (TokenStream, TokenStream, Option<TokenStream>) {
    if depth == 1 {
        let pre = quote! {
            let mut __df_derive_total_leaves: usize = 0;
            for __df_derive_it in items {
                __df_derive_total_leaves += (&(#access)).len();
            }
        };
        return (pre, quote! { __df_derive_total_leaves }, None);
    }
    let pre = quote! {
        let mut __df_derive_total_leaves: usize = 0;
        let mut __df_derive_total_inners: usize = 0;
        for __df_derive_it in items {
            for #outer_bind in (&(#access)).iter() {
                __df_derive_total_leaves += #outer_bind.len();
                __df_derive_total_inners += 1;
            }
        }
    };
    (
        pre,
        quote! { __df_derive_total_leaves },
        Some(quote! { __df_derive_total_inners }),
    )
}

/// `for it in items { for v in (&access).iter() { ... } }` (depth 1) or the
/// triple-nested form (depth 2). Inner-offsets push happens at the
/// inner-loop tail; outer-offsets push at the outer-loop tail.
fn vec_push_loops(
    access: &TokenStream,
    depth: usize,
    outer_bind: &syn::Ident,
    leaf_bind: &syn::Ident,
    per_elem_push: &TokenStream,
    leaf_offsets_post_push: &TokenStream,
) -> TokenStream {
    let inner_offsets = format_ident!("__df_derive_inner_offsets");
    if depth == 1 {
        return quote! {
            for __df_derive_it in items {
                for #leaf_bind in (&(#access)).iter() {
                    #per_elem_push
                }
                #inner_offsets.push(#leaf_offsets_post_push as i64);
            }
        };
    }
    let outer_offsets = format_ident!("__df_derive_outer_offsets");
    quote! {
        for __df_derive_it in items {
            for #outer_bind in (&(#access)).iter() {
                for #leaf_bind in #outer_bind.iter() {
                    #per_elem_push
                }
                #inner_offsets.push(#leaf_offsets_post_push as i64);
            }
            #outer_offsets.push((#inner_offsets.len() - 1) as i64);
        }
    }
}

/// Offsets vec allocation. At depth 1 there's only `inner_offsets`
/// (one per outer row, sized `items.len() + 1`). At depth 2 we add
/// `outer_offsets` sized the same way; `inner_offsets` is sized by
/// the precounted mid-level list count + 1.
fn vec_offsets_decls(depth: usize, inner_offsets_cap: Option<&TokenStream>) -> TokenStream {
    let inner_offsets = format_ident!("__df_derive_inner_offsets");
    if depth == 1 {
        return quote! {
            let mut #inner_offsets: ::std::vec::Vec<i64> =
                ::std::vec::Vec::with_capacity(items.len() + 1);
            #inner_offsets.push(0);
        };
    }
    let cap = inner_offsets_cap.expect("depth==2 sets inner_offsets capacity");
    let outer_offsets = format_ident!("__df_derive_outer_offsets");
    quote! {
        let mut #inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(#cap + 1);
        #inner_offsets.push(0);
        let mut #outer_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #outer_offsets.push(0);
    }
}

/// Stack one or two `LargeListArray::new` layers and route the outer one
/// through `__df_derive_assemble_list_series_unchecked`. The helper
/// consumes a single `LargeListArray` and a leaf logical dtype, and itself
/// wraps that dtype in one `List<>` layer to construct the schema dtype.
/// So at depth 1 the helper's input dtype is the bare leaf logical dtype
/// (e.g. `DataType::Int32`); at depth 2 it is `DataType::List(Box::new(leaf))`.
fn vec_final_assemble(
    depth: usize,
    leaf_dtype_tokens: &TokenStream,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let inner_offsets = format_ident!("__df_derive_inner_offsets");
    if depth == 1 {
        return quote! {
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
                #leaf_dtype_tokens,
            )
        };
    }
    let outer_offsets = format_ident!("__df_derive_outer_offsets");
    let inner_logical_dtype = quote! {
        #pp::DataType::List(::std::boxed::Box::new(#leaf_dtype_tokens))
    };
    quote! {
        let __df_derive_inner_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#inner_offsets)?;
        let __df_derive_inner_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_leaf_arr).clone(),
            ),
            __df_derive_inner_offsets_buf,
            ::std::boxed::Box::new(__df_derive_leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        let __df_derive_outer_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#outer_offsets)?;
        let __df_derive_outer_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_list_arr).clone(),
            ),
            __df_derive_outer_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #inner_logical_dtype,
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
        VecLeafSpec::Bool => quote! { __df_derive_leaf_idx },
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
    }
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
                    __df_derive_flat.push({ #value_expr });
                }
                ::std::option::Option::None => {
                    __df_derive_flat.push(<#native as ::std::default::Default>::default());
                    __df_derive_validity.set(__df_derive_flat.len() - 1, false);
                }
            }
        }
    } else {
        quote! {
            __df_derive_flat.push({ #value_expr });
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
            __df_derive_leaf_arr
        }
    } else {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::from_vec(__df_derive_flat);
            __df_derive_leaf_arr
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
            __df_derive_leaf_arr
        }
    } else {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::Utf8ViewArray = __df_derive_view_buf.freeze();
            __df_derive_leaf_arr
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
        __df_derive_leaf_arr
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
/// `finish_series` that just references the local (with the columnar-context
/// rename applied via `with_name(name)` — the vec-anyvalues context passes
/// `name = ""`, so the rename is a no-op there).
fn vec_encoder(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    depth: usize,
    has_inner_option: bool,
    leaf_dtype: &TokenStream,
) -> Encoder {
    let series_local = vec_encoder_series_local(ctx.idx);
    let name = ctx.name;
    let decl = vec_emit_decl(ctx, spec, depth, has_inner_option, leaf_dtype);
    let finish_series = quote! { #series_local.with_name(#name.into()) };
    Encoder {
        decls: vec![decl],
        push: TokenStream::new(),
        option_push: None,
        finish_series,
        kind: LeafKind::PerElementPush,
        offset_depth: depth,
    }
}

/// Bare-bool variant of the vec encoder: `Vec<bool>` (depth 1, no inner-Option)
/// uses `BooleanArray::from_slice`; `Vec<Vec<bool>>` (depth 2, no inner-Option)
/// uses a bit-packed `MutableBitmap` populated by `set`. Matches the legacy
/// `try_gen_vec_bool_emit` and `try_gen_vec_vec_bool_emit` shapes
/// byte-for-byte.
fn vec_encoder_bool_bare(ctx: &LeafCtx<'_>, depth: usize) -> Encoder {
    let pa_root = super::polars_paths::polars_arrow_root();
    let pp = super::polars_paths::prelude();
    let series_local = vec_encoder_series_local(ctx.idx);
    let body = if depth == 1 {
        bool_bare_depth1_body(ctx.access, &pa_root, &pp)
    } else {
        bool_bare_depth2_body(ctx.access, &pa_root, &pp)
    };
    let name = ctx.name;
    let decl = quote! {
        let #series_local: #pp::Series = { #body };
    };
    Encoder {
        decls: vec![decl],
        push: TokenStream::new(),
        option_push: None,
        finish_series: quote! { #series_local.with_name(#name.into()) },
        kind: LeafKind::PerElementPush,
        offset_depth: depth,
    }
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

/// `Vec<Vec<bool>>` body: bit-packed values bitmap pre-filled `false` with
/// random-access `set(idx, true)` per element. Bench 12 (`Vec<bool>`) and
/// the `Vec<Vec<bool>>` regression matrix both lock in this layout.
fn bool_bare_depth2_body(
    access: &TokenStream,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let leaf_bind = format_ident!("__df_derive_v");
    let outer_bind = format_ident!("__df_derive_inner");
    let inner_offsets = format_ident!("__df_derive_inner_offsets");
    let outer_offsets = format_ident!("__df_derive_outer_offsets");
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    let inner_logical_dtype = quote! { #pp::DataType::List(::std::boxed::Box::new(#leaf_dtype)) };
    quote! {
        let mut __df_derive_total_leaves: usize = 0;
        let mut __df_derive_total_inners: usize = 0;
        for __df_derive_it in items {
            for #outer_bind in (&(#access)).iter() {
                __df_derive_total_leaves += #outer_bind.len();
                __df_derive_total_inners += 1;
            }
        }
        let mut __df_derive_values: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total_leaves);
            __df_derive_b.extend_constant(__df_derive_total_leaves, false);
            __df_derive_b
        };
        let mut #inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(__df_derive_total_inners + 1);
        #inner_offsets.push(0);
        let mut #outer_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #outer_offsets.push(0);
        let mut __df_derive_leaf_idx: usize = 0;
        for __df_derive_it in items {
            for #outer_bind in (&(#access)).iter() {
                for #leaf_bind in #outer_bind.iter() {
                    if *#leaf_bind {
                        __df_derive_values.set(__df_derive_leaf_idx, true);
                    }
                    __df_derive_leaf_idx += 1;
                }
                #inner_offsets.push(__df_derive_leaf_idx as i64);
            }
            #outer_offsets.push((#inner_offsets.len() - 1) as i64);
        }
        let __df_derive_leaf_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(__df_derive_values),
                ::std::option::Option::None,
            );
        let __df_derive_inner_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#inner_offsets)?;
        let __df_derive_inner_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_leaf_arr).clone(),
            ),
            __df_derive_inner_offsets_buf,
            ::std::boxed::Box::new(__df_derive_leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        let __df_derive_outer_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#outer_offsets)?;
        let __df_derive_outer_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_list_arr).clone(),
            ),
            __df_derive_outer_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #inner_logical_dtype,
        )
    }
}

/// `Vec<...>` (`as_str` transform) — same MBVA-based encoder as the bare
/// `String` path, but the value expression sources `&str` via UFCS through
/// `AsRef<str>`. The bytes are copied into the view array once, identical
/// to the `String::as_str()` path.
fn vec_encoder_as_str(
    ctx: &LeafCtx<'_>,
    depth: usize,
    has_inner_option: bool,
    base: &BaseType,
) -> Encoder {
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
    vec_encoder(ctx, &spec, depth, has_inner_option, &leaf_dtype)
}

// --- Top-level dispatcher ---

/// Returns `Some(encoder)` when the (base, transform, wrappers) triple can be
/// served by this encoder IR. As of Step 2 this covers `[]`, `[Option]`,
/// `[Vec]`, `[Vec, Option]`, `[Vec, Vec]`, and `[Vec, Vec, Option]` shapes
/// for every primitive leaf except bare `ISize`/`USize`. Other shapes
/// (`[Option, Vec]`, `[Option, Vec, Option]`, `[Vec, Option, Vec]`, deeper
/// nestings) fall through to the legacy paths in `primitive.rs`.
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
    match wrappers {
        [] => build_leaf(base, transform, ctx),
        [Wrapper::Option] => {
            let leaf = build_leaf(base, transform, ctx)?;
            Some(wrap_option(base, transform, leaf, ctx))
        }
        [Wrapper::Vec] => try_build_vec_encoder(base, transform, ctx, 1, false),
        [Wrapper::Vec, Wrapper::Option] => try_build_vec_encoder(base, transform, ctx, 1, true),
        [Wrapper::Vec, Wrapper::Vec] => try_build_vec_encoder(base, transform, ctx, 2, false),
        [Wrapper::Vec, Wrapper::Vec, Wrapper::Option] => {
            try_build_vec_encoder(base, transform, ctx, 2, true)
        }
        _ => None,
    }
}

/// Build the `vec(inner)` (and `vec(vec(inner))`) encoder for the
/// `(base, transform)` combinations the encoder IR covers.
///
/// Matches `build_leaf`'s coverage: bare numeric, `String`, `Bool`,
/// `Decimal` (with `DecimalToInt128`), `DateTime` (with `DateTimeToInt`),
/// `as_str` borrow, and `to_string`. Returns `None` for `ISize`/`USize`
/// (legacy generic path) and for transforms that don't apply to the base.
fn try_build_vec_encoder(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    ctx: &LeafCtx<'_>,
    depth: usize,
    has_inner_option: bool,
) -> Option<Encoder> {
    match transform {
        None => try_build_vec_encoder_bare(base, ctx, depth, has_inner_option),
        Some(PrimitiveTransform::DateTimeToInt(unit)) => match base {
            BaseType::DateTimeUtc => {
                Some(vec_encoder_datetime(ctx, *unit, depth, has_inner_option))
            }
            _ => None,
        },
        Some(PrimitiveTransform::DecimalToInt128 { precision, scale }) => match base {
            BaseType::Decimal => Some(vec_encoder_decimal(
                ctx,
                *precision,
                *scale,
                depth,
                has_inner_option,
            )),
            _ => None,
        },
        Some(PrimitiveTransform::ToString) => {
            Some(vec_encoder_to_string(ctx, depth, has_inner_option))
        }
        // `as_str` borrow path: same MBVA-based encoder as `String`, but
        // the value expression goes through UFCS (`AsRef<str>`) instead of
        // `String::as_str`. Bytes are copied into the view array once.
        Some(PrimitiveTransform::AsStr) => {
            Some(vec_encoder_as_str(ctx, depth, has_inner_option, base))
        }
    }
}

fn try_build_vec_encoder_bare(
    base: &BaseType,
    ctx: &LeafCtx<'_>,
    depth: usize,
    has_inner_option: bool,
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
            Some(vec_encoder(
                ctx,
                &spec,
                depth,
                has_inner_option,
                &info.dtype,
            ))
        }
        BaseType::String => {
            let pp = super::polars_paths::prelude();
            let leaf_dtype = quote! { #pp::DataType::String };
            let spec = VecLeafSpec::StringLike {
                value_expr: quote! { __df_derive_v.as_str() },
                extra_decls: Vec::new(),
            };
            Some(vec_encoder(
                ctx,
                &spec,
                depth,
                has_inner_option,
                &leaf_dtype,
            ))
        }
        BaseType::Bool => {
            if has_inner_option {
                let pp = super::polars_paths::prelude();
                let leaf_dtype = quote! { #pp::DataType::Boolean };
                Some(vec_encoder(
                    ctx,
                    &VecLeafSpec::Bool,
                    depth,
                    has_inner_option,
                    &leaf_dtype,
                ))
            } else {
                Some(vec_encoder_bool_bare(ctx, depth))
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

fn vec_encoder_datetime(
    ctx: &LeafCtx<'_>,
    unit: DateTimeUnit,
    depth: usize,
    has_inner_option: bool,
) -> Encoder {
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
    vec_encoder(ctx, &spec, depth, has_inner_option, &leaf_dtype)
}

fn vec_encoder_decimal(
    ctx: &LeafCtx<'_>,
    precision: u8,
    scale: u8,
    depth: usize,
    has_inner_option: bool,
) -> Encoder {
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
    vec_encoder(ctx, &spec, depth, has_inner_option, &leaf_dtype)
}

fn vec_encoder_to_string(ctx: &LeafCtx<'_>, depth: usize, has_inner_option: bool) -> Encoder {
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
    vec_encoder(ctx, &spec, depth, has_inner_option, &leaf_dtype)
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
