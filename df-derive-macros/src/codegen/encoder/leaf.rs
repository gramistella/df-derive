//! Primitive leaf builders + shared decl helpers.
//!
//! Each `*_leaf` here returns a single [`LeafArm`] selected by an explicit
//! [`LeafArmKind`]: `Bare` for the unwrapped shape, `Option` for the
//! `[Option]` shape. The dispatcher in [`super::mod`] selects `Bare` for
//! `WrapperShape::Leaf { option_layers: 0 }` and `Option` for
//! `option_layers == 1`. Deeper or smart-pointer-interleaved Option stacks
//! reuse the `Option` arm after collapsing the access into a single
//! `Option<&T>`; Copy leaves materialize that as `Option<T>` with `.copied()`
//! first because their push bodies consume the value in pattern position.
//! Polars folds every nested None into one validity bit.

use crate::ir::{DateTimeUnit, DurationSource, LeafSpec, NumericKind, StringyBase};
use proc_macro2::TokenStream;
use quote::quote;

use super::LeafCtx;
use super::idents;

/// Selects which arm a leaf builder emits. The split lets each leaf
/// override the option-shape per-row work (e.g. the bool 3-arm match
/// against a pre-filled values bitmap, the string-like MBVA + bitmap pair,
/// the bool inner-Option bit-packed values bitmap) without leaking a
/// runtime "is-this-supplied?" check into the dispatcher.
#[derive(Clone, Copy)]
pub(super) enum LeafArmKind {
    /// Unwrapped shape — `WrapperShape::Leaf { option_layers: 0 }`.
    Bare,
    /// `[Option]` shape — `WrapperShape::Leaf { option_layers: 1 }`, plus
    /// the multi-Option leaf wrapper after access collapse.
    Option,
}

/// One arm (bare or `[Option]`) emitted by a `*_leaf` builder. `decls` is
/// emitted once before the per-row loop; `push` is spliced inside the loop;
/// `series` is an expression that evaluates to a `polars::prelude::Series`
/// after the loop.
///
/// Distinct from [`crate::ir::LeafSpec`] (which classifies the unwrapped
/// element type at parse time). This is the encoder-internal product of the
/// dispatch over a `LeafSpec` plus a `LeafCtx` plus a [`LeafArmKind`].
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
    let pa_root = crate::codegen::external_paths::polars_arrow_root();
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
    }
}

/// `let mut #ident: MutableBitmap = MutableBitmap pre-filled with #value over #capacity;`
pub(super) fn mb_decl_filled(
    ident: &syn::Ident,
    capacity: &TokenStream,
    value: bool,
) -> TokenStream {
    let pa_root = crate::codegen::external_paths::polars_arrow_root();
    let b = idents::bitmap_builder();
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap = {
            let mut #b = #pa_root::bitmap::MutableBitmap::with_capacity(#capacity);
            #b.extend_constant(#capacity, #value);
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
    let pa_root = crate::codegen::external_paths::polars_arrow_root();
    quote! {
        let mut #buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(items.len());
    }
}

/// `let mut #buf: MutableBinaryViewArray<[u8]> = MutableBinaryViewArray::<[u8]>::with_capacity(items.len());`
/// — the byte-blob analogue of [`mbva_decl`]. Used by the `Binary` leaf
/// (`#[df_derive(as_binary)]` over `Vec<u8>`) to build a `BinaryView` column
/// without round-tripping through a `Vec<&[u8]>` intermediate.
pub(super) fn mbva_bytes_decl(buf: &syn::Ident) -> TokenStream {
    let pa_root = crate::codegen::external_paths::polars_arrow_root();
    quote! {
        let mut #buf: #pa_root::array::MutableBinaryViewArray<[u8]> =
            #pa_root::array::MutableBinaryViewArray::<[u8]>::with_capacity(items.len());
    }
}

/// Convert a `MutableBitmap` validity buffer into the `Option<Bitmap>`
/// `with_chunk` / `with_validity` arms expect. `MutableBitmap -> Option<Bitmap>`
/// collapses to `None` when no bits are unset, preserving the no-null fast path.
pub(super) fn validity_into_option(validity: &syn::Ident) -> TokenStream {
    let pa_root = crate::codegen::external_paths::polars_arrow_root();
    quote! {
        ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
            #validity,
        )
    }
}

/// Build a Series via `into_series(StringChunked::with_chunk(name, arr))`.
pub(super) fn string_chunked_series(name: &str, arr_expr: &TokenStream) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    quote! {
        #pp::IntoSeries::into_series(
            #pp::StringChunked::with_chunk(#name.into(), { #arr_expr }),
        )
    }
}

/// Build a Series via `into_series(BinaryChunked::with_chunk(name, arr))`.
/// Byte-blob analogue of [`string_chunked_series`].
pub(super) fn binary_chunked_series(name: &str, arr_expr: &TokenStream) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    quote! {
        #pp::IntoSeries::into_series(
            #pp::BinaryChunked::with_chunk(#name.into(), { #arr_expr }),
        )
    }
}

/// Build a Series via `<Series as NamedFrom<_, _>>::new(name.into(), &buf)`.
/// The `.into()` and the `&` borrow are part of the emitted shape and are
/// preserved by callers that splice this into a larger expression.
pub(super) fn named_from_buf(name: &str, buf: &syn::Ident) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) }
}

// --- Leaf builders ---

/// Numeric primitive leaf — covers fixed-width (`i8/.../f64`) and the
/// platform-sized widened variants (`ISize`/`USize` widened to `i64`/`u64`
/// at the leaf push site). Bare arm: `Vec<#native>` storage + `<Chunked>::from_vec`,
/// which consumes the Vec without copying. Option arm: `Vec<#native>` + parallel
/// `MutableBitmap` + `PrimitiveArray::new`. When `info.widen_from` is `Some`,
/// the bare push reads the field as `(#access) as #target` and the
/// validity-arm `Some` push extracts via `(*v) as #target`.
pub(super) fn numeric_leaf(ctx: &LeafCtx<'_>, kind: NumericKind, arm: LeafArmKind) -> LeafArm {
    let info = crate::codegen::type_registry::numeric_info_for(kind);
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let native = &info.native;
    let chunked = &info.chunked;
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = crate::codegen::external_paths::prelude();
    let pa_root = crate::codegen::external_paths::polars_arrow_root();

    match arm {
        LeafArmKind::Bare => {
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
            let bare_value = if kind.is_nonzero() || info.widen_from.is_some() {
                crate::codegen::type_registry::numeric_stored_value(
                    kind,
                    quote! { #access },
                    native,
                )
            } else {
                quote! { (#access).clone() }
            };
            let bare_push = quote! { #buf.push({ #bare_value }); };
            let bare_series = quote! {
                #pp::IntoSeries::into_series(#chunked::from_vec(#name.into(), #buf))
            };
            LeafArm {
                decls: vec![vec_decl(&buf, native)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option => {
            // `Some` arm pushes the value (validity pre-filled to `true` is wrong —
            // we use push-based MutableBitmap here, no pre-fill); `None` arm pushes
            // `<#native>::default()` and `validity.push(false)`. Splitting value vs
            // validity into independent pushes lets the compiler vectorize cleanly.
            // Non-trivial option/smart-pointer access chains are collapsed
            // before this leaf arm is invoked, so the old single
            // `Option<numeric>` token shape remains intact here.
            let v = idents::leaf_value();
            let some_push_value =
                crate::codegen::type_registry::numeric_stored_value(kind, quote! { #v }, native);
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
            LeafArm {
                decls: vec![vec_decl(&buf, native), mb_decl(&validity)],
                push: option_push,
                series: option_series,
            }
        }
    }
}

/// `String` leaf. Bare arm: `MutableBinaryViewArray<str>` accumulator —
/// bypasses the `Vec<&str>` round-trip and the second walk
/// `Series::new(&Vec<&str>)` would do via `from_slice_values`. Option arm:
/// MBVA + parallel `MutableBitmap` (pre-filled `true`) + row counter; `Some`
/// pushes the borrowed `&str` (no validity work), `None` pushes "" and flips
/// a single bit via the safe `MutableBitmap::set`.
pub(super) fn string_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let row_idx = idents::primitive_row_idx(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;

    match arm {
        LeafArmKind::Bare => {
            let bare_push = quote! { #buf.push_value_ignore_validity((#access).as_str()); };
            let bare_series = string_chunked_series(name, &quote! { #buf.freeze() });
            LeafArm {
                decls: vec![mbva_decl(&buf)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option => {
            let v = idents::leaf_value();
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
            LeafArm {
                decls: vec![
                    mbva_decl(&buf),
                    mb_decl_filled(&validity, &quote! { items.len() }, true),
                    row_idx_decl(&row_idx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

/// `Binary` leaf — `#[df_derive(as_binary)]` over a `Vec<u8>` shape. Bare
/// arm: `MutableBinaryViewArray<[u8]>` accumulator, mirrors the `String`
/// path's MBVA bypass (no `Vec<&[u8]>` round-trip). Option arm: MBVA +
/// parallel `MutableBitmap` (pre-filled `true`) + row counter; `Some`
/// pushes the borrowed `&[u8]` (no validity work), `None` pushes an empty
/// slice and flips a single bit via `MutableBitmap::set`.
pub(super) fn binary_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let row_idx = idents::primitive_row_idx(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;

    match arm {
        LeafArmKind::Bare => {
            let bytes = bytes_ref_expr(&quote! { &(#access) });
            let bare_push = quote! { #buf.push_value_ignore_validity(#bytes); };
            let bare_series = binary_chunked_series(name, &quote! { #buf.freeze() });
            LeafArm {
                decls: vec![mbva_bytes_decl(&buf)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option => {
            let v = idents::leaf_value();
            let empty = quote! { &[][..] };
            let bytes = bytes_ref_expr(&quote! { #v });
            let option_push = quote! {
                match &(#access) {
                    ::std::option::Option::Some(#v) => {
                        #buf.push_value_ignore_validity(#bytes);
                    }
                    ::std::option::Option::None => {
                        #buf.push_value_ignore_validity(#empty);
                        #validity.set(#row_idx, false);
                    }
                }
                #row_idx += 1;
            };
            let valid_opt = validity_into_option(&validity);
            let option_series =
                binary_chunked_series(name, &quote! { #buf.freeze().with_validity(#valid_opt) });
            LeafArm {
                decls: vec![
                    mbva_bytes_decl(&buf),
                    mb_decl_filled(&validity, &quote! { items.len() }, true),
                    row_idx_decl(&row_idx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

fn bytes_ref_expr(binding: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote! { ::core::convert::AsRef::<[u8]>::as_ref(#binding) }
}

/// `bool` leaf. Bare arm: `Vec<bool>` + `Series::new`. Keeps the slow path
/// because `BooleanChunked::from_slice` is bulk and faster than
/// `BooleanArray::new` + `with_chunk` for the all-non-null case. Option arm:
/// switches to the bitmap-pair layout (`MutableBitmap` values pre-filled
/// `false` + `MutableBitmap` validity pre-filled `true` + row counter); the
/// 3-arm match makes `Some(false)` zero work, `Some(true)` flips a value
/// bit, `None` flips a validity bit.
pub(super) fn bool_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let row_idx = idents::primitive_row_idx(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = crate::codegen::external_paths::prelude();
    let pa_root = crate::codegen::external_paths::polars_arrow_root();

    match arm {
        LeafArmKind::Bare => {
            let bare_push = quote! { #buf.push({ (#access).clone() }); };
            let bare_series = named_from_buf(name, &buf);
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { bool })],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option => {
            // Non-trivial option/smart-pointer access chains are collapsed
            // before this leaf arm is invoked, so this preserves the legacy
            // single `Option<bool>` match shape.
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
            LeafArm {
                decls: vec![
                    mb_decl_filled(&buf, &quote! { items.len() }, false),
                    mb_decl_filled(&validity, &quote! { items.len() }, true),
                    row_idx_decl(&row_idx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

/// Build push tokens for a `Vec<...>` (or `Vec<Option<...>>`) buffer that
/// holds the result of a per-row mapped expression — used by `Decimal` and
/// `DateTime` leaves which share the same shape (`buf.push({ mapped })` for
/// bare, `match Some/None => Some(mapped)/None` for option). Returns the
/// arm-specific push expression.
fn mapped_push(ctx: &LeafCtx<'_>, leaf: &LeafSpec, arm: LeafArmKind) -> TokenStream {
    let buf = idents::primitive_buf(ctx.base.idx);
    let access = ctx.base.access;
    let decimal_trait = ctx.decimal128_encode_trait;
    match arm {
        LeafArmKind::Bare => {
            let mapped_bare =
                crate::codegen::type_registry::map_primitive_expr(access, leaf, decimal_trait);
            quote! { #buf.push({ #mapped_bare }); }
        }
        LeafArmKind::Option => {
            let v = idents::leaf_value();
            let some_var = quote! { #v };
            let mapped_some =
                crate::codegen::type_registry::map_primitive_expr(&some_var, leaf, decimal_trait);
            quote! {
                match &(#access) {
                    ::std::option::Option::Some(#v) => {
                        #buf.push(::std::option::Option::Some({ #mapped_some }));
                    }
                    ::std::option::Option::None => {
                        #buf.push(::std::option::Option::None);
                    }
                }
            }
        }
    }
}

/// `Decimal` leaf with a `DecimalToInt128` transform. Bare: `Vec<i128>` +
/// `Int128Chunked::from_vec` + `into_decimal_unchecked`. Option: switches to
/// `Vec<Option<i128>>` + `from_iter_options` + `into_decimal_unchecked`.
pub(super) fn decimal_leaf(
    ctx: &LeafCtx<'_>,
    precision: u8,
    scale: u8,
    arm: LeafArmKind,
) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let pp = crate::codegen::external_paths::prelude();
    let int128 = crate::codegen::external_paths::int128_chunked();
    let p = precision as usize;
    let s = scale as usize;
    let leaf = LeafSpec::Decimal { precision, scale };
    let push = mapped_push(ctx, &leaf, arm);
    match arm {
        LeafArmKind::Bare => {
            let bare_series = quote! {{
                let ca = #int128::from_vec(#name.into(), #buf);
                #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
            }};
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { i128 })],
                push,
                series: bare_series,
            }
        }
        LeafArmKind::Option => {
            let option_series = quote! {{
                let ca = <#int128 as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    #name.into(),
                    #buf.into_iter(),
                );
                #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
            }};
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i128> })],
                push,
                series: option_series,
            }
        }
    }
}

/// `DateTime<Tz>` leaf with a `DateTimeToInt(unit)` transform. Bare:
/// `Vec<i64>` + `Series::new` + cast to `Datetime(unit, None)`. Option:
/// switches to `Vec<Option<i64>>` with the same finish path (`Series::new`
/// + cast); only the element type changes.
pub(super) fn datetime_leaf(ctx: &LeafCtx<'_>, unit: DateTimeUnit, arm: LeafArmKind) -> LeafArm {
    mapped_cast_leaf(ctx, &LeafSpec::DateTime(unit), &quote! { i64 }, arm)
}

/// `NaiveDateTime` leaf with a `NaiveDateTimeToInt(unit)` transform. Same
/// storage and dtype as `datetime_leaf`, but maps through `and_utc()` so the
/// naive value is interpreted against the Unix epoch without a timezone.
pub(super) fn naive_datetime_leaf(
    ctx: &LeafCtx<'_>,
    unit: DateTimeUnit,
    arm: LeafArmKind,
) -> LeafArm {
    mapped_cast_leaf(ctx, &LeafSpec::NaiveDateTime(unit), &quote! { i64 }, arm)
}

/// `NaiveDate` leaf — i32 days since 1970-01-01. Bare: `Vec<i32>` +
/// `Series::new` + cast to `Date`. Option: `Vec<Option<i32>>`. Shape
/// matches `datetime_leaf` modulo native type / cast dtype.
pub(super) fn naive_date_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    mapped_cast_leaf(ctx, &LeafSpec::NaiveDate, &quote! { i32 }, arm)
}

/// `NaiveTime` leaf — i64 nanoseconds since midnight. Bare: `Vec<i64>` +
/// `Series::new` + cast to `Time`.
pub(super) fn naive_time_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    mapped_cast_leaf(ctx, &LeafSpec::NaiveTime, &quote! { i64 }, arm)
}

/// `Duration` leaf (std or chrono) — i64 ticks, unit decided at parse time.
/// Bare: `Vec<i64>` + `Series::new` + cast to `Duration(unit)`.
pub(super) fn duration_leaf(
    ctx: &LeafCtx<'_>,
    unit: DateTimeUnit,
    source: DurationSource,
    arm: LeafArmKind,
) -> LeafArm {
    mapped_cast_leaf(
        ctx,
        &LeafSpec::Duration { unit, source },
        &quote! { i64 },
        arm,
    )
}

/// Shared body of every `Vec<#native>` + `Series::new` + `.cast(&dtype)?`
/// leaf — `DateTime`, `NaiveDateTime`, `NaiveDate`, `NaiveTime`, both
/// Duration variants.
/// Each caller passes the leaf spec (drives `mapped_push` and the cast
/// dtype) and the native storage type; arm selection swaps between
/// `Vec<#native>` and `Vec<Option<#native>>`.
fn mapped_cast_leaf(
    ctx: &LeafCtx<'_>,
    leaf: &LeafSpec,
    native: &TokenStream,
    arm: LeafArmKind,
) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let push = mapped_push(ctx, leaf, arm);
    let dtype = leaf.dtype();
    let series_new = named_from_buf(name, &buf);
    let series_finish = quote! {{
        let mut s = #series_new;
        s = s.cast(&#dtype)?;
        s
    }};
    match arm {
        LeafArmKind::Bare => LeafArm {
            decls: vec![vec_decl(&buf, native)],
            push,
            series: series_finish,
        },
        LeafArmKind::Option => LeafArm {
            decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<#native> })],
            push,
            series: series_finish,
        },
    }
}

/// `as_string` (Display) leaf. Reused `String` scratch + MBVA accumulator —
/// each row clears the scratch, runs `Display::fmt` into it, then pushes the
/// resulting `&str` to the view array (which copies the bytes). Option arm
/// adds the validity bitmap pair on top of the same MBVA + scratch layout.
pub(super) fn as_string_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let scratch = idents::primitive_str_scratch(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let row_idx = idents::primitive_row_idx(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let scratch_decl =
        quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); };

    match arm {
        LeafArmKind::Bare => {
            let bare_push = quote! {
                {
                    use ::std::fmt::Write as _;
                    #scratch.clear();
                    ::std::write!(&mut #scratch, "{}", &(#access)).unwrap();
                    #buf.push_value_ignore_validity(#scratch.as_str());
                }
            };
            let bare_series = string_chunked_series(name, &quote! { #buf.freeze() });
            LeafArm {
                decls: vec![mbva_decl(&buf), scratch_decl],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option => {
            let v = idents::leaf_value();
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
            // Option-arm decl ordering matches the prior shared
            // `option_for_string_like` emission: MBVA first, then the `as_string`
            // scratch as an "extra decl", then the validity bitmap and row
            // counter. `as_string` has a `String` scratch on top of the MBVA
            // pair and we preserve that ordering here.
            LeafArm {
                decls: vec![
                    mbva_decl(&buf),
                    scratch_decl,
                    mb_decl_filled(&validity, &quote! { items.len() }, true),
                    row_idx_decl(&row_idx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

/// `as_str` (borrowed) leaf. `Vec<&str>` (or `Vec<Option<&str>>` in option
/// context) borrows from `items`. `StringyBase` carries the type-path
/// information (`String`, the field's struct ident, or a generic-parameter
/// ident) and lets the bare-`String` deref-coercion path stay distinct from
/// the UFCS path — both are produced by [`super::stringy_value_expr`].
pub(super) fn as_str_leaf(ctx: &LeafCtx<'_>, base: &StringyBase, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let series_finish = named_from_buf(name, &buf);
    match arm {
        LeafArmKind::Bare => {
            let bare_value = super::stringy_value_expr(base, access, super::StringyExprKind::Bare);
            let bare_push = quote! { #buf.push(#bare_value); };
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { &str })],
                push: bare_push,
                series: series_finish,
            }
        }
        LeafArmKind::Option => {
            let option_value =
                super::stringy_value_expr(base, access, super::StringyExprKind::OptionDeref);
            let option_push = quote! { #buf.push(#option_value); };
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<&str> })],
                push: option_push,
                series: series_finish,
            }
        }
    }
}
