//! Option-wrapping logic.
//!
//! The `option_for_*` helpers turn a bare-leaf `Encoder` into an `[Option]`
//! shape by switching the storage layout (e.g. `Vec<T>` →
//! `Vec<Option<T>>`, MBVA → MBVA + validity bitmap, `Vec<bool>` →
//! bit-packed values bitmap + validity bitmap), reusing the leaf's
//! `option_push` body for the per-row work.
//!
//! `wrap_option` dispatches per [`LeafShape`] to the right helper, and
//! `wrap_multi_option_*` handles `Option<…<Option<T>>>` stacks of depth ≥ 2
//! by collapsing the access into a single `Option<&T>` (Polars folds every
//! nested None into one validity bit).

use crate::ir::{BaseType, PrimitiveTransform};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents;
use super::leaf::{
    mb_decl, mb_decl_filled, mbva_decl, row_idx_decl, validity_into_option, vec_decl,
};
use super::{Encoder, LeafCtx, LeafShape, StringyBase, collapse_options_to_ref};

/// Destructure an inner `Encoder::Leaf` produced by a leaf builder. The
/// option combinators only ever wrap leaf encoders (multi-column nested
/// encoders take their own dedicated paths), so any other variant is a
/// programmer error.
fn unwrap_leaf(inner: Encoder) -> (Vec<TokenStream>, TokenStream, Option<TokenStream>) {
    match inner {
        Encoder::Leaf {
            decls,
            push,
            option_push,
            series: _,
        } => (decls, push, option_push),
        Encoder::Multi { .. } => {
            unreachable!("option combinator received a multi-column encoder")
        }
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
    let buf = idents::primitive_buf(ctx.idx);
    let validity = idents::primitive_validity(ctx.idx);
    let row_idx = idents::primitive_row_idx(ctx.idx);
    let name = ctx.name;
    let valid_opt = validity_into_option(&validity);
    let mut decls = vec![mbva_decl(&buf)];
    decls.extend(extra_decls);
    decls.push(mb_decl_filled(&validity, true));
    decls.push(row_idx_decl(&row_idx));
    let arr_expr = quote! { #buf.freeze().with_validity(#valid_opt) };
    let (_, _, inner_option_push) = unwrap_leaf(inner);
    Encoder::Leaf {
        decls,
        push: inner_option_push.expect("string-like leaf must supply option_push"),
        option_push: None,
        series: super::leaf::string_chunked_series(name, &arr_expr),
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
    let buf = idents::primitive_buf(ctx.idx);
    let validity = idents::primitive_validity(ctx.idx);
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
    let (_, inner_push, inner_option_push) = unwrap_leaf(inner);
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, native), mb_decl(&validity)],
        push: inner_option_push.unwrap_or(inner_push),
        option_push: None,
        series: finish_series,
    }
}

/// Option combinator for the bool leaf: bit-packed values bitmap (pre-filled
/// `false`) + validity bitmap (pre-filled `true`) + row counter. Finishes
/// through `BooleanArray::new` + `BooleanChunked::with_chunk`.
fn option_for_bool(ctx: &LeafCtx<'_>, inner: Encoder) -> Encoder {
    let buf = idents::primitive_buf(ctx.idx);
    let validity = idents::primitive_validity(ctx.idx);
    let row_idx = idents::primitive_row_idx(ctx.idx);
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
    let (_, _, inner_option_push) = unwrap_leaf(inner);
    Encoder::Leaf {
        decls: vec![
            mb_decl_filled(&buf, false),
            mb_decl_filled(&validity, true),
            row_idx_decl(&row_idx),
        ],
        push: inner_option_push.expect("bool_leaf must supply option_push"),
        option_push: None,
        series: finish_series,
    }
}

/// Option combinator for Decimal: switches to `Vec<Option<i128>>` and
/// `from_iter_options`-based finish.
fn option_for_decimal(ctx: &LeafCtx<'_>, precision: u8, scale: u8, inner: Encoder) -> Encoder {
    let buf = idents::primitive_buf(ctx.idx);
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
    let (_, _, inner_option_push) = unwrap_leaf(inner);
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i128> })],
        push: inner_option_push.expect("decimal_leaf must supply option_push"),
        option_push: None,
        series: finish_series,
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
    let buf = idents::primitive_buf(ctx.idx);
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
    let (_, _, inner_option_push) = unwrap_leaf(inner);
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i64> })],
        push: inner_option_push.expect("datetime_leaf must supply option_push"),
        option_push: None,
        series: finish_series,
    }
}

/// Option combinator for `as_str`: switches to `Vec<Option<&str>>` storage.
/// The finish path is `Series::new(name, &buf)` regardless.
fn option_for_as_str(ctx: &LeafCtx<'_>, inner: Encoder) -> Encoder {
    let buf = idents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    let finish_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    let (_, _, inner_option_push) = unwrap_leaf(inner);
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<&str> })],
        push: inner_option_push.expect("as_str_leaf must supply option_push"),
        option_push: None,
        series: finish_series,
    }
}

pub(super) fn wrap_option(shape: &LeafShape<'_>, inner: Encoder, ctx: &LeafCtx<'_>) -> Encoder {
    match shape {
        LeafShape::Numeric(base) => option_for_numeric(ctx, base, inner),
        LeafShape::NumericWidened(base) => option_for_numeric_widened(ctx, base, inner),
        LeafShape::String => option_for_string_like(ctx, vec![], inner),
        LeafShape::Bool => option_for_bool(ctx, inner),
        LeafShape::DateTime(unit) => option_for_datetime(ctx, *unit, inner),
        LeafShape::Decimal { precision, scale } => {
            option_for_decimal(ctx, *precision, *scale, inner)
        }
        LeafShape::AsString => {
            // `as_string` has a `String` scratch on top of the MBVA pair —
            // pass it as an extra decl into the shared option-string-like
            // combinator so the layout exactly matches the prior emission.
            let scratch = idents::primitive_str_scratch(ctx.idx);
            let extra = vec![
                quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); },
            ];
            option_for_string_like(ctx, extra, inner)
        }
        LeafShape::AsStr(_) => option_for_as_str(ctx, inner),
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
    shape: &LeafShape<'_>,
    ctx: &LeafCtx<'_>,
    layers: usize,
) -> Encoder {
    debug_assert!(layers >= 2);
    if let LeafShape::AsStr(stringy) = shape {
        return wrap_multi_option_as_str(stringy, ctx, layers);
    }
    let orig_access = ctx.access.clone();
    let local = idents::multi_option_local(ctx.idx);
    let local_access = quote! { #local };
    let collapsed_chain = collapse_options_to_ref(&orig_access, layers);
    // Copy-eligible primitives (numeric, `ISize`/`USize`, `Bool`) flatten
    // through `.copied()`; everything else through `.cloned()`. The local
    // shadows the field for the inner option-leaf machinery so its existing
    // `match #access { Some(v) => ... }` push body just works.
    let materializer = if is_copy_leaf_shape(shape) {
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
    let leaf = super::vec::build_leaf(shape, &new_ctx);
    let wrapped = wrap_option(shape, leaf, &new_ctx);
    match wrapped {
        Encoder::Leaf {
            decls,
            push,
            option_push,
            series,
        } => Encoder::Leaf {
            decls,
            push: quote! {
                #setup
                #push
            },
            option_push,
            series,
        },
        Encoder::Multi { .. } => {
            unreachable!("wrap_option over a primitive leaf must yield a Leaf")
        }
    }
}

/// `as_str`-specific multi-Option wrapper. Builds the same `Vec<Option<&str>>`
/// buffer + finish as `option_for_as_str`, but the per-row push collapses the
/// stacked `Option`s into a single `Option<&str>` borrowed from the original
/// field — the buffer's borrow needs to live for the whole pass, which a
/// per-row local owning `String` cannot provide.
fn wrap_multi_option_as_str(base: &StringyBase<'_>, ctx: &LeafCtx<'_>, layers: usize) -> Encoder {
    let buf = idents::primitive_buf(ctx.idx);
    let name = ctx.name;
    let pp = crate::codegen::polars_paths::prelude();
    let collapsed_ref = collapse_options_to_ref(ctx.access, layers);
    let push = if base.is_string() {
        // For `String` base, `&String` deref-coerces to `&str`, so the
        // collapsed `Option<&String>` maps to `Option<&str>` directly via
        // `String::as_str`.
        quote! { #buf.push((#collapsed_ref).map(::std::string::String::as_str)); }
    } else {
        let ty_path = base.ty_path();
        quote! {
            #buf.push(
                (#collapsed_ref).map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
            );
        }
    };
    let finish_series = quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) };
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<&str> })],
        push,
        option_push: None,
        series: finish_series,
    }
}

/// `Copy` test for the multi-Option per-row materializer. Numeric leaves,
/// `ISize`/`USize`, and `Bool` are `Copy`; `String`, `DateTime`, `Decimal`,
/// `as_string`, and the `as_str` borrow path are not (and `as_str` takes its
/// own branch above before reaching this helper).
const fn is_copy_leaf_shape(shape: &LeafShape<'_>) -> bool {
    matches!(
        shape,
        LeafShape::Numeric(_) | LeafShape::NumericWidened(_) | LeafShape::Bool
    )
}
