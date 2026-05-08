//! Encoder IR: a compositional encoder model for per-field `DataFrame` columnization.
//!
//! Each leaf encoder knows how to emit (decls, push, finish) for one base type.
//! The `option(inner)` combinator wraps a leaf to add `Option<...>` semantics.
//! `vec(inner)` adds an arbitrary-depth `LargeListArray` stack over the leaf.
//! Per-field codegen folds the wrapper stack right-to-left over the leaf to
//! assemble the final emission.
//!
//! Each leaf builder emits a single `LeafArm` selected by an explicit
//! [`leaf::LeafArmKind`] — `Bare` for the unwrapped shape, `Option` for the
//! `[Option]` shape. The split lets the `bool` leaf override the option case
//! with a 3-arm match (so `Some(false)` is a true no-op against a values
//! bitmap pre-filled with `false`) and lets each leaf pick the right
//! buffer/finish layout for its option semantics without leaking a runtime
//! "is-this-supplied?" check into the dispatcher.
//!
//! `Vec`-bearing wrappers are normalized into a [`crate::ir::VecLayers`]
//! (one entry per `Vec` layer; each entry tracks whether an outer `Option`
//! adjoins it as list-level validity) **at the parser**. The encoder
//! destructures the IR's `WrapperShape` directly without re-normalizing.
//! After normalization the encoder emits an N-deep precount, an N-deep push
//! loop, an N-deep stack of `LargeListArray::new` calls — one flat values
//! buffer at the deepest layer, one optional inner validity bitmap at the
//! deepest layer, one optional outer-list validity bitmap per `Vec` layer.
//! Polars folds consecutive `Option` layers into a single validity bit, so
//! `[Option, Option]` collapses to a single bit and `[Option, Option, Vec]`
//! collapses the leading two `Option`s into one bit on the `Vec` layer —
//! the runtime semantics match because the only observable null is whichever
//! bit is the outermost Polars validity.

mod emit;
pub(in crate::codegen) mod idents;
mod leaf;
mod leaf_kind;
mod nested;
mod option;
mod shape_walk;
mod tuple;
mod vec;

use crate::ir::{LeafSpec, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub use nested::{NestedLeafCtx, build_nested_encoder};
pub use tuple::{
    build_field_emit as build_tuple_field_emit,
    build_field_entries as build_tuple_field_entries,
};

/// Build the type-as-path token stream for a struct/generic field. For a
/// struct referenced without args (e.g. `Address`) this is the bare ident;
/// for a struct referenced with args (e.g. `Foo<M>` or `Foo<M, N>`) it is
/// the turbofish form `Foo::<M, N>`, valid in both expression and type
/// position. Generic type parameters use the bare ident (the macro injects
/// the trait bounds that make `T::method()` resolve).
pub fn build_type_path(
    ident: &Ident,
    args: Option<&syn::AngleBracketedGenericArguments>,
) -> TokenStream {
    args.map_or_else(
        || quote! { #ident },
        |ab| {
            let inner = &ab.args;
            quote! { #ident::<#inner> }
        },
    )
}

/// Token stream for the type-as-path expression used in UFCS calls
/// (`<#ty as AsRef<str>>::as_ref(...)`). The `String` base maps to
/// `::std::string::String`; structs/generics splice through
/// `build_type_path`.
pub(super) fn stringy_base_ty_path(base: &crate::ir::StringyBase) -> TokenStream {
    match base {
        crate::ir::StringyBase::String => quote! { ::std::string::String },
        crate::ir::StringyBase::Struct(ident, args) => build_type_path(ident, args.as_ref()),
        crate::ir::StringyBase::Generic(ident) => quote! { #ident },
    }
}

/// Which `StringyBase` value-expression shape the caller needs. The `as_str`
/// leaf path produces `&str` / `Option<&str>` from a field access in three
/// distinct contexts (bare leaf push, single-Option leaf push, multi-Option
/// collapsed push) and the vec(`as_str`) path produces a `&str` from a
/// per-row binding inside the leaf push body. Each shape branches on
/// `StringyBase::is_string()` the same way — `String` deref-coerces or uses
/// `String::as_str`, non-`String` bases UFCS through `<T as AsRef<str>>`.
pub(super) enum StringyExprKind {
    /// Leaf push for `WrapperShape::Leaf { option_layers: 0 }`. Materializes
    /// `&str`-coerceable from a field access. `String`: `&(#binding)` (relies
    /// on `&String -> &str` deref-coercion). Non-`String`: UFCS
    /// `<TyPath as AsRef<str>>::as_ref(&(#binding))`.
    Bare,
    /// Leaf push for `WrapperShape::Leaf { option_layers: 1 }`. Materializes
    /// `Option<&str>` from a `&Option<T>` field access. `String`:
    /// `(#binding).as_deref()`. Non-`String`:
    /// `(#binding).as_ref().map(<TyPath as AsRef<str>>::as_ref)`.
    OptionDeref,
    /// Multi-Option leaf push (`option_layers >= 2`) — operates on a
    /// pre-collapsed `Option<&T>` binding. `String`:
    /// `(#binding).map(String::as_str)`. Non-`String`:
    /// `(#binding).map(<TyPath as AsRef<str>>::as_ref)`.
    CollapsedOption,
    /// Vec(`as_str`) per-row leaf push body. The binding is `&T` (loop
    /// variable). `String`: `#binding.as_str()`. Non-`String`:
    /// `<TyPath as AsRef<str>>::as_ref(#binding)`.
    MbvaValue,
}

/// Build the value expression for an `as_str`-style leaf at a given call
/// site. Centralizes the `is_string()` branch shared by the leaf builder
/// (`leaf::as_str_leaf`), the multi-Option wrapper
/// (`option::wrap_multi_option_as_str`), and the vec combinator
/// (`vec::vec_encoder_as_str`). See [`StringyExprKind`] for the four
/// shapes the helper produces.
pub(super) fn stringy_value_expr(
    base: &crate::ir::StringyBase,
    binding: &TokenStream,
    kind: StringyExprKind,
) -> TokenStream {
    let is_string = base.is_string();
    match kind {
        StringyExprKind::Bare => {
            if is_string {
                quote! { &(#binding) }
            } else {
                let ty_path = stringy_base_ty_path(base);
                quote! { <#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#binding)) }
            }
        }
        StringyExprKind::OptionDeref => {
            if is_string {
                quote! { (#binding).as_deref() }
            } else {
                let ty_path = stringy_base_ty_path(base);
                quote! { (#binding).as_ref().map(<#ty_path as ::core::convert::AsRef<str>>::as_ref) }
            }
        }
        StringyExprKind::CollapsedOption => {
            if is_string {
                quote! { (#binding).map(::std::string::String::as_str) }
            } else {
                let ty_path = stringy_base_ty_path(base);
                quote! { (#binding).map(<#ty_path as ::core::convert::AsRef<str>>::as_ref) }
            }
        }
        StringyExprKind::MbvaValue => {
            if is_string {
                quote! { #binding.as_str() }
            } else {
                let ty_path = stringy_base_ty_path(base);
                quote! { <#ty_path as ::core::convert::AsRef<str>>::as_ref(#binding) }
            }
        }
    }
}

/// Build an expression that collapses `n` `Option` layers above a base
/// expression into a single `Option<&Inner>`. `base` must already be a
/// reference (or place expression that auto-derefs to a reference). For
/// `n == 0` this is a no-op (returns the base unchanged).
pub(super) fn collapse_options_to_ref(base: &TokenStream, n: usize) -> TokenStream {
    if n == 0 {
        return base.clone();
    }
    let param = idents::collapse_option_param();
    let mut out = quote! { (#base).as_ref() };
    for _ in 1..n {
        out = quote! { #out.and_then(|#param| #param.as_ref()) };
    }
    out
}

/// Per-field encoder state. Variant determines whether the encoder serves
/// the single-column primitive path (`Leaf`) or the multi-column nested
/// path (`Multi`); the field set is type-enforced per variant.
pub enum Encoder {
    /// Single-column primitive leaf path. `decls` is emitted once before the
    /// per-row loop; `push` is spliced inside the loop; `series` is an
    /// expression that evaluates to a `polars::prelude::Series` after the
    /// loop, which the caller wraps as `columns.push(series.into())`.
    Leaf {
        decls: Vec<TokenStream>,
        push: TokenStream,
        series: TokenStream,
    },
    /// Multi-column nested path. The `columnar` block executes once after
    /// the call site's per-row loop and pushes one Series per inner schema
    /// column of the nested type onto the outer `columns` vec directly.
    Multi { columnar: TokenStream },
}

/// Per-leaf identity shared across both the primitive ([`LeafCtx`]) and
/// nested ([`NestedLeafCtx`]) encoder paths: the per-row access expression,
/// the field's stable index (used to namespace generated idents), and the
/// field's column name.
pub struct BaseCtx<'a> {
    pub access: &'a TokenStream,
    pub idx: usize,
    pub name: &'a str,
}

/// Per-leaf metadata threaded into the leaf builders.
pub struct LeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub decimal128_encode_trait: &'a TokenStream,
    /// Smart-pointer layers (`Box`/`Rc`/`Arc`/`Cow`) sitting between a
    /// wrapper (`Option`/`Vec`) and the leaf type. The leaf push body
    /// applies this many `*` derefs to the per-element binding so primitive
    /// pushes see the inner value, not the smart pointer. Most leaves work
    /// without explicit derefs (method-call autoderef handles `arc.as_str()`
    /// etc.) but pattern-binding-by-value sites need the unwrap.
    pub inner_smart_ptr_depth: usize,
}

// --- Top-level dispatcher ---

/// Build the encoder for a primitive-routed field. Covers every wrapper
/// stack the parser accepts for every primitive leaf — bare numerics,
/// `ISize`/`USize` (widened to `i64`/`u64` at the codegen boundary),
/// `String`, `Bool`, `Decimal`, `DateTime`, `as_str`, `to_string`, plus
/// `Option<…<Option<T>>>` stacks of arbitrary depth (Polars folds
/// consecutive `Option`s into a single validity bit, so the encoder
/// collapses the access expression to a single `Option<&T>` before invoking
/// the option-leaf push) and every primitive vec-bearing shape. The
/// function is total on parser-validated IR.
pub fn build_encoder(leaf: &LeafSpec, wrapper: &WrapperShape, ctx: &LeafCtx<'_>) -> Encoder {
    match wrapper {
        WrapperShape::Leaf { option_layers: 0 } => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(leaf, ctx, leaf::LeafArmKind::Bare);
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        WrapperShape::Leaf { option_layers: 1 } => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(leaf, ctx, leaf::LeafArmKind::Option);
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        // Primitive multi-Option leaf shapes (`Option<Option<T>>`,
        // `Option<Option<Option<T>>>`, …): pre-collapse the access to
        // `Option<&T>` (Polars folds every nested None to one validity bit)
        // and feed it through the single-Option leaf machinery via a
        // synthesized per-row local. Nested multi-Option (`Option<Option<T>>`
        // over a struct/generic) is handled separately in `build_nested_encoder`.
        WrapperShape::Leaf {
            option_layers: layers,
        } => option::wrap_multi_option_primitive(leaf, ctx, *layers),
        WrapperShape::Vec(vec_layers) => vec::try_build_vec_encoder(leaf, ctx, vec_layers),
    }
}
