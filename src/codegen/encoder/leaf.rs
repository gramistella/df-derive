//! Primitive leaf builders + shared decl helpers.
//!
//! Each `*_leaf` here returns a [`LeafBuilder`] carrying both the bare-leaf
//! and `[Option]` shape pieces (decls, push, finish). The dispatcher in
//! [`super::mod`] selects `spec.bare` for `WrapperShape::Leaf { option_layers:
//! 0 }` and `spec.option` for `option_layers == 1`. Deeper Option stacks
//! reuse `spec.option` after collapsing the access into a single
//! `Option<&T>` (Polars folds every nested None into one validity bit).

use crate::ir::{DateTimeUnit, LeafSpec, NumericKind, StringyBase};
use proc_macro2::TokenStream;
use quote::quote;

use super::LeafCtx;
use super::idents;

/// Per-leaf bundle of token streams. Every primitive leaf carries both a
/// bare-shape [`LeafArm`] and an `[Option]`-shape [`LeafArm`]. The split lets
/// each leaf override the option-shape per-row work (e.g. the bool 3-arm
/// match against a pre-filled values bitmap, the string-like MBVA + bitmap
/// pair, the bool inner-Option bit-packed values bitmap) without leaking a
/// runtime "is-this-supplied?" check into the dispatcher.
///
/// Distinct from [`crate::ir::LeafSpec`] (which classifies the unwrapped
/// element type at parse time). This builder bundle is the encoder-internal
/// product of the dispatch over a `LeafSpec` plus a `LeafCtx`.
pub(super) struct LeafBuilder {
    pub bare: LeafArm,
    pub option: LeafArm,
}

/// One arm (bare or `[Option]`) of a [`LeafBuilder`]. `decls` is emitted once
/// before the per-row loop; `push` is spliced inside the loop; `series` is
/// an expression that evaluates to a `polars::prelude::Series` after the
/// loop.
pub(super) struct LeafArm {
    pub decls: Vec<TokenStream>,
    pub push: TokenStream,
    pub series: TokenStream,
}

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
    let b = idents::bitmap_builder();
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap = {
            let mut #b = #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
            #b.extend_constant(items.len(), #value);
            #b
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

/// Numeric primitive leaf — covers fixed-width (`i8/.../f64`) and the
/// platform-sized widened variants (`ISize`/`USize` widened to `i64`/`u64`
/// at the leaf push site). Bare arm: `Vec<#native>` storage + `<Chunked>::from_vec`,
/// which consumes the Vec without copying. Option arm: `Vec<#native>` + parallel
/// `MutableBitmap` + `PrimitiveArray::new`. When `info.widen_from` is `Some`,
/// the bare push reads the field as `(#access) as #target` and the
/// validity-arm `Some` push extracts via `(*v) as #target`.
pub(super) fn numeric_leaf(ctx: &LeafCtx<'_>, kind: NumericKind) -> LeafBuilder {
    let info = crate::codegen::type_registry::numeric_info_for(kind);
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let native = &info.native;
    let chunked = &info.chunked;
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = crate::codegen::polars_paths::prelude();
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();

    // Wrap the cloned access expression in `{ ... }` to match the legacy
    // primitive emitter's exact token shape. The block wrap is a syntactic
    // no-op (the expression evaluates identically), but the legacy
    // `try_gen_*` path emitted it and benches like `01_top_level_vec` and
    // `vec_vec_i32` are sensitive to the resulting MIR shape — emitting
    // `push(x.clone())` instead of `push({ x.clone() })` reproducibly
    // regresses these tight loops by 5-12% even though rustc/LLVM should
    // see equivalent MIR. Match the legacy shape exactly.
    //
    // Widening (`isize`/`usize` → `i64`/`u64`) casts the field expression to
    // `info.native` (the storage type), not to `info.widen_from` (the
    // source type) — `widen_from.is_some()` is just the gating signal.
    let bare_value = if info.widen_from.is_some() {
        quote! { ((#access) as #native) }
    } else {
        quote! { (#access).clone() }
    };
    let bare_push = quote! { #buf.push({ #bare_value }); };
    let bare_series = quote! {
        #pp::IntoSeries::into_series(#chunked::from_vec(#name.into(), #buf))
    };
    // `Some` arm pushes the value (validity pre-filled to `true` is wrong —
    // we use push-based MutableBitmap here, no pre-fill); `None` arm pushes
    // `<#native>::default()` and `validity.push(false)`. Splitting value vs
    // validity into independent pushes lets the compiler vectorize cleanly.
    let v = idents::leaf_value();
    let some_push_value = if info.widen_from.is_some() {
        quote! { (#v as #native) }
    } else {
        quote! { #v }
    };
    let option_push = quote! {
        match #access {
            ::std::option::Option::Some(#v) => {
                #buf.push(#some_push_value);
                #validity.push(true);
            }
            ::std::option::Option::None => {
                #buf.push(<#native as ::std::default::Default>::default());
                #validity.push(false);
            }
        }
    };
    let valid_opt = validity_into_option(&validity);
    let option_series = quote! {{
        let arr = #pa_root::array::PrimitiveArray::<#native>::new(
            <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
            #buf.into(),
            #valid_opt,
        );
        #pp::IntoSeries::into_series(#chunked::with_chunk(#name.into(), arr))
    }};
    LeafBuilder {
        bare: LeafArm {
            decls: vec![vec_decl(&buf, native)],
            push: bare_push,
            series: bare_series,
        },
        option: LeafArm {
            decls: vec![vec_decl(&buf, native), mb_decl(&validity)],
            push: option_push,
            series: option_series,
        },
    }
}

/// `String` leaf. Bare arm: `MutableBinaryViewArray<str>` accumulator —
/// bypasses the `Vec<&str>` round-trip and the second walk
/// `Series::new(&Vec<&str>)` would do via `from_slice_values`. Option arm:
/// MBVA + parallel `MutableBitmap` (pre-filled `true`) + row counter; `Some`
/// pushes the borrowed `&str` (no validity work), `None` pushes "" and flips
/// a single bit via the safe `MutableBitmap::set`.
pub(super) fn string_leaf(ctx: &LeafCtx<'_>) -> LeafBuilder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let row_idx = idents::primitive_row_idx(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;

    let v = idents::leaf_value();
    let bare_push = quote! { #buf.push_value_ignore_validity((#access).as_str()); };
    let bare_series = string_chunked_series(name, &quote! { #buf.freeze() });
    let option_push = quote! {
        match &(#access) {
            ::std::option::Option::Some(#v) => {
                #buf.push_value_ignore_validity(#v.as_str());
            }
            ::std::option::Option::None => {
                #buf.push_value_ignore_validity("");
                #validity.set(#row_idx, false);
            }
        }
        #row_idx += 1;
    };
    let valid_opt = validity_into_option(&validity);
    let option_series =
        string_chunked_series(name, &quote! { #buf.freeze().with_validity(#valid_opt) });
    LeafBuilder {
        bare: LeafArm {
            decls: vec![mbva_decl(&buf)],
            push: bare_push,
            series: bare_series,
        },
        option: LeafArm {
            decls: vec![
                mbva_decl(&buf),
                mb_decl_filled(&validity, true),
                row_idx_decl(&row_idx),
            ],
            push: option_push,
            series: option_series,
        },
    }
}

/// `bool` leaf. Bare arm: `Vec<bool>` + `Series::new`. Keeps the slow path
/// because `BooleanChunked::from_slice` is bulk and faster than
/// `BooleanArray::new` + `with_chunk` for the all-non-null case. Option arm:
/// switches to the bitmap-pair layout (`MutableBitmap` values pre-filled
/// `false` + `MutableBitmap` validity pre-filled `true` + row counter); the
/// 3-arm match makes `Some(false)` zero work, `Some(true)` flips a value
/// bit, `None` flips a validity bit.
pub(super) fn bool_leaf(ctx: &LeafCtx<'_>) -> LeafBuilder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let row_idx = idents::primitive_row_idx(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = crate::codegen::polars_paths::prelude();
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();

    let bare_push = quote! { #buf.push({ (#access).clone() }); };
    let bare_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    let option_push = quote! {
        match (#access) {
            ::std::option::Option::Some(true) => { #buf.set(#row_idx, true); }
            ::std::option::Option::Some(false) => {}
            ::std::option::Option::None => { #validity.set(#row_idx, false); }
        }
        #row_idx += 1;
    };
    let valid_opt = validity_into_option(&validity);
    let option_series = quote! {{
        let arr = #pa_root::array::BooleanArray::new(
            #pa_root::datatypes::ArrowDataType::Boolean,
            ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#buf),
            #valid_opt,
        );
        #pp::IntoSeries::into_series(
            #pp::BooleanChunked::with_chunk(#name.into(), arr),
        )
    }};
    LeafBuilder {
        bare: LeafArm {
            decls: vec![vec_decl(&buf, &quote! { bool })],
            push: bare_push,
            series: bare_series,
        },
        option: LeafArm {
            decls: vec![
                mb_decl_filled(&buf, false),
                mb_decl_filled(&validity, true),
                row_idx_decl(&row_idx),
            ],
            push: option_push,
            series: option_series,
        },
    }
}

/// Build push tokens for a `Vec<...>` (or `Vec<Option<...>>`) buffer that
/// holds the result of a per-row mapped expression — used by `Decimal` and
/// `DateTime` leaves which share the same shape (`buf.push({ mapped })` for
/// bare, `match Some/None => Some(mapped)/None` for option).
fn mapped_push_pair(ctx: &LeafCtx<'_>, leaf: &LeafSpec) -> (TokenStream, TokenStream) {
    let buf = idents::primitive_buf(ctx.base.idx);
    let access = ctx.base.access;
    let decimal_trait = ctx.decimal128_encode_trait;
    let v = idents::leaf_value();
    let mapped_bare =
        crate::codegen::type_registry::map_primitive_expr(access, leaf, decimal_trait);
    let mapped_some = {
        let some_var = quote! { #v };
        crate::codegen::type_registry::map_primitive_expr(&some_var, leaf, decimal_trait)
    };
    let bare_push = quote! { #buf.push({ #mapped_bare }); };
    let option_push = quote! {
        match &(#access) {
            ::std::option::Option::Some(#v) => {
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
pub(super) fn decimal_leaf(ctx: &LeafCtx<'_>, precision: u8, scale: u8) -> LeafBuilder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let pp = crate::codegen::polars_paths::prelude();
    let int128 = crate::codegen::polars_paths::int128_chunked();
    let p = precision as usize;
    let s = scale as usize;
    let leaf = LeafSpec::Decimal { precision, scale };
    let (bare_push, option_push) = mapped_push_pair(ctx, &leaf);
    let bare_series = quote! {{
        let ca = #int128::from_vec(#name.into(), #buf);
        #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
    }};
    let option_series = quote! {{
        let ca = <#int128 as #pp::NewChunkedArray<_, _>>::from_iter_options(
            #name.into(),
            #buf.into_iter(),
        );
        #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
    }};
    LeafBuilder {
        bare: LeafArm {
            decls: vec![vec_decl(&buf, &quote! { i128 })],
            push: bare_push,
            series: bare_series,
        },
        option: LeafArm {
            decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i128> })],
            push: option_push,
            series: option_series,
        },
    }
}

/// `DateTime<Utc>` leaf with a `DateTimeToInt(unit)` transform. Bare:
/// `Vec<i64>` + `Series::new` + cast to `Datetime(unit, None)`. Option:
/// switches to `Vec<Option<i64>>` with the same finish path (`Series::new`
/// + cast); only the element type changes.
pub(super) fn datetime_leaf(ctx: &LeafCtx<'_>, unit: DateTimeUnit) -> LeafBuilder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let pp = crate::codegen::polars_paths::prelude();
    let leaf = LeafSpec::DateTime(unit);
    let (bare_push, option_push) = mapped_push_pair(ctx, &leaf);
    let dtype = leaf.dtype();
    let series_finish = quote! {{
        let mut s = <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf);
        s = s.cast(&#dtype)?;
        s
    }};
    LeafBuilder {
        bare: LeafArm {
            decls: vec![vec_decl(&buf, &quote! { i64 })],
            push: bare_push,
            series: series_finish.clone(),
        },
        option: LeafArm {
            decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i64> })],
            push: option_push,
            series: series_finish,
        },
    }
}

/// `as_string` (Display) leaf. Reused `String` scratch + MBVA accumulator —
/// each row clears the scratch, runs `Display::fmt` into it, then pushes the
/// resulting `&str` to the view array (which copies the bytes). Option arm
/// adds the validity bitmap pair on top of the same MBVA + scratch layout.
pub(super) fn as_string_leaf(ctx: &LeafCtx<'_>) -> LeafBuilder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let scratch = idents::primitive_str_scratch(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let row_idx = idents::primitive_row_idx(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;

    let v = idents::leaf_value();
    let bare_push = quote! {
        {
            use ::std::fmt::Write as _;
            #scratch.clear();
            ::std::write!(&mut #scratch, "{}", &(#access)).unwrap();
            #buf.push_value_ignore_validity(#scratch.as_str());
        }
    };
    let bare_series = string_chunked_series(name, &quote! { #buf.freeze() });
    let option_push = quote! {
        match &(#access) {
            ::std::option::Option::Some(#v) => {
                use ::std::fmt::Write as _;
                #scratch.clear();
                ::std::write!(&mut #scratch, "{}", #v).unwrap();
                #buf.push_value_ignore_validity(#scratch.as_str());
            }
            ::std::option::Option::None => {
                #buf.push_value_ignore_validity("");
                #validity.set(#row_idx, false);
            }
        }
        #row_idx += 1;
    };
    let valid_opt = validity_into_option(&validity);
    let option_series =
        string_chunked_series(name, &quote! { #buf.freeze().with_validity(#valid_opt) });
    let scratch_decl =
        quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); };
    LeafBuilder {
        bare: LeafArm {
            decls: vec![mbva_decl(&buf), scratch_decl.clone()],
            push: bare_push,
            series: bare_series,
        },
        // Option-arm decl ordering matches the prior shared
        // `option_for_string_like` emission: MBVA first, then the `as_string`
        // scratch as an "extra decl", then the validity bitmap and row
        // counter. `as_string` has a `String` scratch on top of the MBVA
        // pair and we preserve that ordering here.
        option: LeafArm {
            decls: vec![
                mbva_decl(&buf),
                scratch_decl,
                mb_decl_filled(&validity, true),
                row_idx_decl(&row_idx),
            ],
            push: option_push,
            series: option_series,
        },
    }
}

/// `as_str` (borrowed) leaf. `Vec<&str>` (or `Vec<Option<&str>>` in option
/// context) borrows from `items`. `StringyBase` carries the type-path
/// information (`String`, the field's struct ident, or a generic-parameter
/// ident) and lets the bare-`String` deref-coercion path stay distinct from
/// the UFCS path.
pub(super) fn as_str_leaf(ctx: &LeafCtx<'_>, base: &StringyBase) -> LeafBuilder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
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
        let ty_path = super::stringy_base_ty_path(base);
        (
            quote! { #buf.push(<#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#access))); },
            quote! {
                #buf.push(
                    (#access).as_ref().map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
                );
            },
        )
    };
    let series_finish = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    LeafBuilder {
        bare: LeafArm {
            decls: vec![vec_decl(&buf, &quote! { &str })],
            push: bare_push,
            series: series_finish.clone(),
        },
        option: LeafArm {
            decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<&str> })],
            push: option_push,
            series: series_finish,
        },
    }
}
