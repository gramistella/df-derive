//! Multi-`Option` leaf wrappers.
//!
//! `wrap_multi_option_primitive` and `wrap_multi_option_as_str` handle the
//! `Option<…<Option<T>>>` stacks of depth ≥ 2 by collapsing the access into
//! a single `Option<&T>` (Polars folds every nested None into one validity
//! bit). The single-Option case is served directly by the leaf's
//! `LeafSpec::option` arm — `mod.rs::build_encoder` selects it without
//! needing a wrapper.

use quote::quote;

use super::idents;
use super::leaf::{LeafArm, LeafSpec, vec_decl};
use super::{BaseCtx, Encoder, LeafCtx, LeafShape, StringyBase, collapse_options_to_ref};

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
    let orig_access = ctx.base.access.clone();
    let local = idents::multi_option_local(ctx.base.idx);
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
        base: BaseCtx {
            access: &local_access,
            idx: ctx.base.idx,
            name: ctx.base.name,
        },
        decimal128_encode_trait: ctx.decimal128_encode_trait,
    };
    let LeafSpec {
        option: LeafArm {
            decls,
            push,
            series,
        },
        ..
    } = super::vec::build_leaf(shape, &new_ctx);
    Encoder::Leaf {
        decls,
        push: quote! {
            #setup
            #push
        },
        series,
    }
}

/// `as_str`-specific multi-Option wrapper. Builds the same `Vec<Option<&str>>`
/// buffer + finish as the single-Option `as_str` arm, but the per-row push
/// collapses the stacked `Option`s into a single `Option<&str>` borrowed
/// from the original field — the buffer's borrow needs to live for the
/// whole pass, which a per-row local owning `String` cannot provide.
fn wrap_multi_option_as_str(base: &StringyBase<'_>, ctx: &LeafCtx<'_>, layers: usize) -> Encoder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let pp = crate::codegen::polars_paths::prelude();
    let collapsed_ref = collapse_options_to_ref(ctx.base.access, layers);
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
