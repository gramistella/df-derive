//! Primitive leaf builders + shared decl helpers.
//!
//! Each `*_leaf` here returns the `Encoder` for the bare-leaf shape (no
//! wrappers); the matching `option_for_*` helpers in [`super::option`] turn
//! that bare encoder into the `[Option]` shape, and the deeper stacks
//! compose through the vec/option combinators.

use crate::ir::{BaseType, DateTimeUnit, PrimitiveTransform};
use proc_macro2::TokenStream;
use quote::quote;

use super::{Encoder, LeafCtx, PopulatorIdents, StringyBase};

// --- Common decl helpers ---

/// `let mut #buf: Vec<#elem> = Vec::with_capacity(items.len());`
pub(super) fn vec_decl(buf: &syn::Ident, elem: &TokenStream) -> TokenStream {
    quote! {
        let mut #buf: ::std::vec::Vec<#elem> =
            ::std::vec::Vec::with_capacity(items.len());
    }
}

/// `let mut #ident: MutableBitmap = MutableBitmap::with_capacity(items.len());`
/// (no pre-fill — push-based use only).
pub(super) fn mb_decl(ident: &syn::Ident) -> TokenStream {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
    }
}

/// `let mut #ident: MutableBitmap = MutableBitmap pre-filled with #value over items.len();`
pub(super) fn mb_decl_filled(ident: &syn::Ident, value: bool) -> TokenStream {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b = #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
            __df_derive_b.extend_constant(items.len(), #value);
            __df_derive_b
        };
    }
}

/// `let mut #ident: usize = 0;` — row counter for the pre-filled-bitmap leaves.
pub(super) fn row_idx_decl(ident: &syn::Ident) -> TokenStream {
    quote! { let mut #ident: usize = 0; }
}

/// `let mut #buf: MutableBinaryViewArray<str> = MutableBinaryViewArray::<str>::with_capacity(items.len());`
pub(super) fn mbva_decl(buf: &syn::Ident) -> TokenStream {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    quote! {
        let mut #buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(items.len());
    }
}

/// Convert a `MutableBitmap` validity buffer into the `Option<Bitmap>`
/// `with_chunk` / `with_validity` arms expect. `MutableBitmap -> Option<Bitmap>`
/// collapses to `None` when no bits are unset, preserving the no-null fast path.
pub(super) fn validity_into_option(validity: &syn::Ident) -> TokenStream {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    quote! {
        ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
            #validity,
        )
    }
}

/// Build a Series via `into_series(StringChunked::with_chunk(name, arr))`.
pub(super) fn string_chunked_series(name: &str, arr_expr: &TokenStream) -> TokenStream {
    let pp = crate::codegen::polars_paths::prelude();
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
pub(super) fn numeric_leaf(ctx: &LeafCtx<'_>, base: &BaseType) -> Encoder {
    let info = crate::codegen::type_registry::numeric_info(base)
        .expect("numeric_leaf must be called with a numeric BaseType");
    numeric_leaf_with_info(ctx, &info, None)
}

/// `ISize`/`USize` widened leaf — Polars cannot represent platform-sized
/// integers natively, so the encoder widens reads to `i64`/`u64` via an
/// `as` cast at every push site. Storage matches `compute_mapping`'s
/// element dtype (`Int64`/`UInt64`) so the downstream chunked-array build
/// produces the schema dtype with no post-finish cast.
pub(super) fn numeric_leaf_widened(ctx: &LeafCtx<'_>, base: &BaseType) -> Encoder {
    let info = crate::codegen::type_registry::isize_usize_widened_info(base)
        .expect("numeric_leaf_widened requires an `ISize`/`USize` BaseType");
    let cast_target = info.native.clone();
    numeric_leaf_with_info(ctx, &info, Some(&cast_target))
}

/// Common builder for both `numeric_leaf` and `numeric_leaf_widened`. When
/// `widen_to` is `Some(target)`, the per-row push reads the field as
/// `(#access) as #target` and the validity-arm `Some` push extracts via
/// `(*v) as #target`. The `None`-arm default and the finish pieces are
/// identical to the bare-numeric path because the storage type already
/// matches the target.
fn numeric_leaf_with_info(
    ctx: &LeafCtx<'_>,
    info: &crate::codegen::type_registry::NumericInfo,
    widen_to: Option<&TokenStream>,
) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let native = &info.native;
    let chunked = &info.chunked;
    let access = ctx.access;
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();

    // Wrap the cloned access expression in `{ ... }` to match the legacy
    // primitive emitter's exact token shape. The block wrap is a syntactic
    // no-op (the expression evaluates identically), but the legacy
    // `try_gen_*` path emitted it and benches like `01_top_level_vec` and
    // `vec_vec_i32` are sensitive to the resulting MIR shape — emitting
    // `push(x.clone())` instead of `push({ x.clone() })` reproducibly
    // regresses these tight loops by 5-12% even though rustc/LLVM should
    // see equivalent MIR. Match the legacy shape exactly.
    let bare_value = widen_to.map_or_else(
        || quote! { (#access).clone() },
        |target| quote! { ((#access) as #target) },
    );
    let bare_push = quote! { #buf.push({ #bare_value }); };
    let finish_series = quote! {
        #pp::IntoSeries::into_series(#chunked::from_vec(#name.into(), #buf))
    };
    // `Some` arm pushes the value (validity pre-filled to `true` is wrong —
    // we use push-based MutableBitmap here, no pre-fill); `None` arm pushes
    // `<#native>::default()` and `validity.push(false)`. Splitting value vs
    // validity into independent pushes lets the compiler vectorize cleanly.
    let some_push_value = widen_to.map_or_else(
        || quote! { __df_derive_v },
        |target| quote! { (__df_derive_v as #target) },
    );
    let option_push = quote! {
        match #access {
            ::std::option::Option::Some(__df_derive_v) => {
                #buf.push(#some_push_value);
                #validity.push(true);
            }
            ::std::option::Option::None => {
                #buf.push(<#native as ::std::default::Default>::default());
                #validity.push(false);
            }
        }
    };
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, native)],
        push: bare_push,
        option_push: Some(option_push),
        series: finish_series,
    }
}

/// Bare `String` leaf — accumulates straight into a
/// `MutableBinaryViewArray<str>` buffer. Bypasses the `Vec<&str>` round-trip
/// and the second walk `Series::new(&Vec<&str>)` would do via `from_slice_values`.
pub(super) fn string_leaf(ctx: &LeafCtx<'_>) -> Encoder {
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
    Encoder::Leaf {
        decls: vec![mbva_decl(&buf)],
        push: bare_push,
        option_push: Some(option_push),
        series: finish_series,
    }
}

/// Bare `bool` leaf — `Vec<bool>` + `Series::new`. Keeps the slow path because
/// `BooleanChunked::from_slice` is bulk and faster than `BooleanArray::new` +
/// `with_chunk` for the all-non-null case.
pub(super) fn bool_leaf(ctx: &LeafCtx<'_>) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let row_idx = PopulatorIdents::primitive_row_idx(ctx.idx);
    let access = ctx.access;
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();

    let bare_push = quote! { #buf.push({ (#access).clone() }); };
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
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { bool })],
        push: bare_push,
        option_push: Some(option_push),
        series: finish_series,
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
        crate::codegen::type_registry::map_primitive_expr(access, Some(transform), decimal_trait);
    let mapped_some = {
        let some_var = quote! { __df_derive_v };
        crate::codegen::type_registry::map_primitive_expr(&some_var, Some(transform), decimal_trait)
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
pub(super) fn decimal_leaf(ctx: &LeafCtx<'_>, precision: u8, scale: u8) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    let int128 = crate::codegen::polars_paths::int128_chunked();
    let p = precision as usize;
    let s = scale as usize;
    let transform = PrimitiveTransform::DecimalToInt128 { precision, scale };
    let (bare_push, option_push) = mapped_push_pair(ctx, &transform);
    let finish_series = quote! {{
        let ca = #int128::from_vec(#name.into(), #buf);
        #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
    }};
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { i128 })],
        push: bare_push,
        option_push: Some(option_push),
        series: finish_series,
    }
}

/// `DateTime<Utc>` leaf with a `DateTimeToInt(unit)` transform. Bare:
/// `Vec<i64>` + `Series::new` + cast to `Datetime(unit, None)`. Option:
/// switches to `Vec<Option<i64>>` with the same finish path.
pub(super) fn datetime_leaf(ctx: &LeafCtx<'_>, unit: DateTimeUnit) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    let transform = PrimitiveTransform::DateTimeToInt(unit);
    let (bare_push, option_push) = mapped_push_pair(ctx, &transform);
    let dtype = crate::codegen::type_registry::compute_full_dtype(
        &BaseType::DateTimeUtc,
        Some(&transform),
        &[],
    );
    let finish_series = quote! {{
        let mut s = <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf);
        s = s.cast(&#dtype)?;
        s
    }};
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { i64 })],
        push: bare_push,
        option_push: Some(option_push),
        series: finish_series,
    }
}

/// `as_string` (Display) leaf. Reused `String` scratch + MBVA accumulator —
/// each row clears the scratch, runs `Display::fmt` into it, then pushes the
/// resulting `&str` to the view array (which copies the bytes).
pub(super) fn as_string_leaf(ctx: &LeafCtx<'_>) -> Encoder {
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
    Encoder::Leaf {
        decls: vec![
            mbva_decl(&buf),
            quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); },
        ],
        push: bare_push,
        option_push: Some(option_push),
        series: finish_series,
    }
}

/// `as_str` (borrowed) leaf. `Vec<&str>` (or `Vec<Option<&str>>` in option
/// context) borrows from `items`. `StringyBase` carries the type-path
/// information (`String`, the field's struct ident, or a generic-parameter
/// ident) and lets the bare-`String` deref-coercion path stay distinct from
/// the UFCS path.
pub(super) fn as_str_leaf(ctx: &LeafCtx<'_>, base: &StringyBase<'_>) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let access = ctx.access;
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    // Bare `String` base (with redundant `as_str`) and `as_str` on a
    // non-string base have different push expressions: `String`'s `&String`
    // deref-coerces to `&str` so the plain `&` form works there; non-string
    // bases need UFCS through the type path.
    let (bare_push, option_push) = if base.is_string() {
        (
            quote! { #buf.push(&(#access)); },
            quote! { #buf.push((#access).as_deref()); },
        )
    } else {
        let ty_path = base.ty_path();
        (
            quote! { #buf.push(<#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#access))); },
            quote! {
                #buf.push(
                    (#access).as_ref().map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
                );
            },
        )
    };
    let finish_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { &str })],
        push: bare_push,
        option_push: Some(option_push),
        series: finish_series,
    }
}
