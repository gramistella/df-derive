//! Option-wrapping logic.
//!
//! The `option_for_*` helpers turn a bare-leaf `Encoder` into an `[Option]`
//! shape by switching the storage layout (e.g. `Vec<T>` →
//! `Vec<Option<T>>`, MBVA → MBVA + validity bitmap, `Vec<bool>` →
//! bit-packed values bitmap + validity bitmap), reusing the leaf's
//! `option_push` body for the per-row work.
//!
//! `wrap_option` dispatches per (base, transform) to the right helper, and
//! `wrap_multi_option_*` handles `Option<…<Option<T>>>` stacks of depth ≥ 2
//! by collapsing the access into a single `Option<&T>` (Polars folds every
//! nested None into one validity bit).

use crate::ir::{BaseType, PrimitiveTransform};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::leaf::{
    mb_decl, mb_decl_filled, mbva_decl, row_idx_decl, validity_into_option, vec_decl,
};
use super::{Encoder, LeafCtx, LeafKind, PopulatorIdents, collapse_options_to_ref};

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
        finish: Encoder::series_finish(super::leaf::string_chunked_series(name, &arr_expr)),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

/// Option combinator for numeric leaves: adds a parallel `MutableBitmap`
/// validity buffer and finishes through `PrimitiveArray::new` +
/// `<Chunked>::with_chunk`.
fn option_for_numeric(ctx: &LeafCtx<'_>, base: &BaseType, inner: Encoder) -> Encoder {
    let info = crate::codegen::type_registry::numeric_info(base)
        .expect("option_for_numeric requires a numeric BaseType");
    option_for_numeric_with_info(ctx, &info, inner)
}

/// Option combinator for the `ISize`/`USize` widened leaves. Identical to
/// `option_for_numeric` plumbing-wise — the only difference lives at the
/// per-row push site, baked into `inner.option_push` by `numeric_leaf_widened`.
fn option_for_numeric_widened(ctx: &LeafCtx<'_>, base: &BaseType, inner: Encoder) -> Encoder {
    let info = crate::codegen::type_registry::isize_usize_widened_info(base)
        .expect("option_for_numeric_widened requires an `ISize`/`USize` BaseType");
    option_for_numeric_with_info(ctx, &info, inner)
}

fn option_for_numeric_with_info(
    ctx: &LeafCtx<'_>,
    info: &crate::codegen::type_registry::NumericInfo,
    inner: Encoder,
) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    let native = &info.native;
    let chunked = &info.chunked;
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
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
        decls: vec![vec_decl(&buf, native), mb_decl(&validity)],
        push: inner.option_push.unwrap_or(inner.push),
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: inner.kind,
        offset_depth: inner.offset_depth,
    }
}

/// Option combinator for the bool leaf: bit-packed values bitmap (pre-filled
/// `false`) + validity bitmap (pre-filled `true`) + row counter. Finishes
/// through `BooleanArray::new` + `BooleanChunked::with_chunk`.
fn option_for_bool(ctx: &LeafCtx<'_>, inner: Encoder) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let validity = PopulatorIdents::primitive_validity(ctx.idx);
    let row_idx = PopulatorIdents::primitive_row_idx(ctx.idx);
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
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

/// Option combinator for Decimal: switches to `Vec<Option<i128>>` and
/// `from_iter_options`-based finish.
fn option_for_decimal(ctx: &LeafCtx<'_>, precision: u8, scale: u8, inner: Encoder) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    let int128 = crate::codegen::polars_paths::int128_chunked();
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

/// Option combinator for `DateTime`: switches to `Vec<Option<i64>>`. The
/// finish path is identical structurally (`Series::new` + cast); only the
/// element type changes — so we can reuse the inner finish.
fn option_for_datetime(
    ctx: &LeafCtx<'_>,
    unit: crate::ir::DateTimeUnit,
    inner: Encoder,
) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    let dtype = crate::codegen::type_registry::compute_full_dtype(
        &BaseType::DateTimeUtc,
        Some(&PrimitiveTransform::DateTimeToInt(unit)),
        &[],
    );
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

/// Option combinator for `as_str`: switches to `Vec<Option<&str>>` storage.
/// The finish path is `Series::new(name, &buf)` regardless.
fn option_for_as_str(ctx: &LeafCtx<'_>, inner: Encoder) -> Encoder {
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
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

pub(super) fn wrap_option(
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
            BaseType::ISize | BaseType::USize => option_for_numeric_widened(ctx, base, inner),
            BaseType::String => option_for_string_like(ctx, vec![], inner),
            BaseType::Bool => option_for_bool(ctx, inner),
            BaseType::DateTimeUtc
            | BaseType::Decimal
            | BaseType::Struct(..)
            | BaseType::Generic(_) => {
                unreachable!("df-derive: wrap_option reached for a leaf shape build_leaf rejects")
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

/// Build the encoder for a primitive leaf with `option_layers >= 2` consecutive
/// `Option`s above it (Polars folds them all to one validity bit). Strategy
/// per leaf-kind:
///
/// - **`as_str` borrow path**: the leaf's owning buffer is
///   `Vec<Option<&str>>` borrowing from `items`. Using a per-row local would
///   discard the borrow at row end, so we collapse the access expression all
///   the way to `Option<&str>` (one shared `as_ref().and_then(...).map(...)`
///   chain) and push it directly. Borrows from the field, lives for the
///   whole pass.
/// - **Owning leaves (numeric, `ISize`/`USize`, `Bool`, `String`, `Decimal`,
///   `DateTime`, `to_string`)**: the buffer holds owned values, so a per-row
///   `Option<T>` local materialised by `.copied()` (Copy types) or
///   `.cloned()` (non-Copy) and fed back through the standard single-Option
///   leaf machinery is sound. The clone is per-row only on this rare slow
///   path; the fast paths still apply for `[]` and `[Option]` shapes.
pub(super) fn wrap_multi_option_primitive(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    ctx: &LeafCtx<'_>,
    layers: usize,
) -> Encoder {
    debug_assert!(layers >= 2);
    if matches!(transform, Some(PrimitiveTransform::AsStr)) {
        return wrap_multi_option_as_str(base, ctx, layers);
    }
    let orig_access = ctx.access.clone();
    let local = format_ident!("__df_derive_mo_{}", ctx.idx);
    let local_access = quote! { #local };
    let collapsed_chain = collapse_options_to_ref(&orig_access, layers);
    // Copy-eligible primitives (numeric, `ISize`/`USize`, `Bool`) flatten
    // through `.copied()`; everything else through `.cloned()`. The local
    // shadows the field for the inner option-leaf machinery so its existing
    // `match #access { Some(v) => ... }` push body just works.
    let materializer = if is_copy_primitive_base(base) {
        quote! { .copied() }
    } else {
        quote! { .cloned() }
    };
    let setup = quote! {
        let #local: ::std::option::Option<_> = #collapsed_chain #materializer;
    };
    let new_ctx = LeafCtx {
        access: &local_access,
        idx: ctx.idx,
        name: ctx.name,
        decimal128_encode_trait: ctx.decimal128_encode_trait,
    };
    let leaf = super::vec::build_leaf(base, transform, &new_ctx);
    let mut wrapped = wrap_option(base, transform, leaf, &new_ctx);
    let original_push = wrapped.push.clone();
    wrapped.push = quote! {
        #setup
        #original_push
    };
    wrapped
}

/// `as_str`-specific multi-Option wrapper. Builds the same `Vec<Option<&str>>`
/// buffer + finish as `option_for_as_str`, but the per-row push collapses the
/// stacked `Option`s into a single `Option<&str>` borrowed from the original
/// field — the buffer's borrow needs to live for the whole pass, which a
/// per-row local owning `String` cannot provide.
fn wrap_multi_option_as_str(base: &BaseType, ctx: &LeafCtx<'_>, layers: usize) -> Encoder {
    let ty_path = match base {
        BaseType::String => quote! { ::std::string::String },
        BaseType::Struct(ident, args) => {
            crate::codegen::strategy::build_type_path(ident, args.as_ref())
        }
        BaseType::Generic(ident) => quote! { #ident },
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
            "df-derive: wrap_multi_option_as_str reached on a non-stringy base \
             (parser rejects `as_str` on these bases)"
        ),
    };
    let buf = PopulatorIdents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    let collapsed_ref = collapse_options_to_ref(ctx.access, layers);
    let push = if matches!(base, BaseType::String) {
        // For `String` base, `&String` deref-coerces to `&str`, so the
        // collapsed `Option<&String>` maps to `Option<&str>` directly via
        // `String::as_str`.
        quote! { #buf.push((#collapsed_ref).map(::std::string::String::as_str)); }
    } else {
        quote! {
            #buf.push(
                (#collapsed_ref).map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
            );
        }
    };
    let finish_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    Encoder {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<&str> })],
        push,
        option_push: None,
        finish: Encoder::series_finish(finish_series),
        kind: LeafKind::PerElementPush,
        offset_depth: 0,
    }
}

/// `Copy` primitive base test for the multi-Option per-row materializer.
/// Numerics, `ISize`/`USize`, and `Bool` are `Copy`; `String`, `DateTime`,
/// `Decimal`, and the struct/generic bases are not.
const fn is_copy_primitive_base(base: &BaseType) -> bool {
    matches!(
        base,
        BaseType::I8
            | BaseType::I16
            | BaseType::I32
            | BaseType::I64
            | BaseType::U8
            | BaseType::U16
            | BaseType::U32
            | BaseType::U64
            | BaseType::F32
            | BaseType::F64
            | BaseType::ISize
            | BaseType::USize
            | BaseType::Bool
    )
}
