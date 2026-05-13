//! Option/access-chain leaf wrappers.
//!
//! `wrap_option_access_chain_primitive` handles `Option` stacks and
//! smart-pointer steps that cannot use the legacy single plain `Option<T>`
//! fast path. It collapses the access into a single optional reference at
//! the exact boundary encoded by `AccessChain` (Polars still folds every
//! nested `None` into one validity bit).

use crate::ir::{AccessChain, LeafSpec, StringyBase};
use quote::quote;

use super::idents;
use super::leaf::{LeafArm, LeafArmKind, named_from_buf, vec_decl};
use super::{BaseCtx, Encoder, LeafCtx, access_chain_to_ref};

/// Build the encoder for a primitive leaf with any non-trivial `AccessChain`
/// above it: multiple `Option`s, smart pointers under an `Option`, or both.
/// Strategy per leaf-kind:
///
/// - **`as_str` borrow path**: the leaf's owning buffer is
///   `Vec<Option<&str>>` borrowing from `items`. Using a per-row local would
///   discard the borrow at row end, so we collapse the access expression all
///   the way to `Option<&str>` (one shared `as_ref().and_then(...).map(...)`
///   chain) and push it directly. Borrows from the field, lives for the
///   whole pass.
/// - **Owning leaves (numeric, `ISize`/`USize`, `Bool`, `String`, `Decimal`,
///   `DateTime`, `to_string`)**: Copy leaves materialise a per-row
///   `Option<T>` with `.copied()` because their single-Option push bodies
///   consume the value in pattern position. Non-Copy leaves keep the
///   collapsed `Option<&T>` and feed that to the same option leaf machinery,
///   whose push bodies already borrow before formatting/encoding/pushing.
pub(super) fn wrap_option_access_chain_primitive(
    leaf: &LeafSpec,
    ctx: &LeafCtx<'_>,
    access: &AccessChain,
    layers: usize,
) -> Encoder {
    debug_assert!(layers >= 1);
    if let LeafSpec::AsStr(stringy) = leaf {
        return wrap_option_access_chain_as_str(stringy, ctx, access);
    }
    let orig_access = ctx.base.access.clone();
    let local = idents::multi_option_local(ctx.base.idx);
    let local_access = quote! { #local };
    let collapsed_chain = super::access_chain_to_option_ref(&quote! { &(#orig_access) }, access);
    // Copy-eligible primitives (numeric, `ISize`/`USize`, `Bool`, copy-like
    // chrono leaves) flatten through `.copied()`. Non-Copy leaves stay as
    // `Option<&T>` so this path does not synthesize a hidden `T: Clone`
    // bound for display, decimal, datetime, binary, or string leaves.
    let setup = if leaf.is_copy() {
        quote! {
            let #local: ::std::option::Option<_> = #collapsed_chain.copied();
        }
    } else {
        quote! {
            let #local: ::std::option::Option<_> = #collapsed_chain;
        }
    };
    let new_ctx = LeafCtx {
        base: BaseCtx {
            access: &local_access,
            idx: ctx.base.idx,
            name: ctx.base.name,
        },
        decimal128_encode_trait: ctx.decimal128_encode_trait,
    };
    let LeafArm {
        decls,
        push,
        series,
    } = super::vec::build_leaf(leaf, &new_ctx, LeafArmKind::Option);
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
/// whole pass, which a per-row local owning `String` cannot provide. The
/// `String`-vs-UFCS branch is shared with the single-Option leaf and the
/// vec(`as_str`) path through [`super::stringy_value_expr`].
fn wrap_option_access_chain_as_str(
    base: &StringyBase,
    ctx: &LeafCtx<'_>,
    access: &AccessChain,
) -> Encoder {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let orig_access = ctx.base.access;
    let collapsed_ref = access_chain_to_ref(&quote! { &(#orig_access) }, access).expr;
    let value = super::stringy_value_expr(
        base,
        &collapsed_ref,
        super::StringyExprKind::CollapsedOption,
    );
    let push = quote! { #buf.push(#value); };
    let finish_series = named_from_buf(name, &buf);
    Encoder::Leaf {
        decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<&str> })],
        push,
        series: finish_series,
    }
}
