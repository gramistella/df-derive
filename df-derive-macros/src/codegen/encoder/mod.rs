//! Per-field encoder construction.
//!
//! `Vec` wrappers are already normalized into [`crate::ir::VecLayers`] before
//! this phase. Polars folds consecutive `Option` layers into a single validity
//! bit, so the encoder collapses multi-Option access chains before emitting
//! the option leaf arm.

mod emit;
pub(in crate::codegen) mod idents;
mod leaf;
mod leaf_kind;
mod nested_columns;
mod nested_leaf;
mod option;
mod shape_walk;
mod tuple;
mod vec;

use crate::ir::{AccessChain, AccessStep, LeafShape, PrimitiveLeaf, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;
use syn::PathArguments;

use super::external_paths::ExternalPaths;

pub use nested_leaf::{NestedLeafCtx, build_nested_encoder};
pub use tuple::{
    build_field_emit as build_tuple_field_emit, build_field_entries as build_tuple_field_entries,
};

/// Build type tokens that are valid in both type and associated-call positions.
pub fn struct_type_tokens(ty: &syn::Type) -> TokenStream {
    if let syn::Type::Path(type_path) = ty
        && type_path.qself.is_none()
    {
        let mut path = type_path.path.clone();
        if let Some(segment) = path.segments.last_mut()
            && let PathArguments::AngleBracketed(args) = &mut segment.arguments
        {
            args.colon2_token.get_or_insert_with(Default::default);
        }
        return quote! { #path };
    }
    quote! { #ty }
}

pub(super) fn stringy_base_ty_path(base: &crate::ir::StringyBase) -> TokenStream {
    match base {
        crate::ir::StringyBase::String => quote! { ::std::string::String },
        crate::ir::StringyBase::BorrowedStr => quote! { &'_ str },
        crate::ir::StringyBase::CowStr => quote! { ::std::borrow::Cow<'_, str> },
        crate::ir::StringyBase::Struct(ty) => struct_type_tokens(ty),
        crate::ir::StringyBase::Generic(ident) => quote! { #ident },
    }
}

#[derive(Clone, Copy)]
pub(super) enum StringyExprKind {
    Bare,
    /// Uses closures so smart-pointer deref coercions fire below `Option`.
    OptionDeref,
    CollapsedOption,
    MbvaValue,
}

pub(super) fn stringy_value_expr(
    base: &crate::ir::StringyBase,
    binding: &TokenStream,
    kind: StringyExprKind,
) -> TokenStream {
    if matches!(
        base,
        crate::ir::StringyBase::BorrowedStr | crate::ir::StringyBase::CowStr
    ) {
        let v = idents::leaf_value();
        return match kind {
            StringyExprKind::Bare => {
                quote! { ::core::convert::AsRef::<str>::as_ref(&(#binding)) }
            }
            StringyExprKind::OptionDeref => {
                quote! {
                    (#binding).as_ref().map(|#v| {
                        ::core::convert::AsRef::<str>::as_ref(#v)
                    })
                }
            }
            StringyExprKind::CollapsedOption => {
                quote! {
                    (#binding).map(|#v| {
                        ::core::convert::AsRef::<str>::as_ref(#v)
                    })
                }
            }
            StringyExprKind::MbvaValue => {
                quote! { ::core::convert::AsRef::<str>::as_ref(#binding) }
            }
        };
    }
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
            let v = idents::leaf_value();
            if is_string {
                quote! { (#binding).as_ref().map(|#v| #v.as_str()) }
            } else {
                let ty_path = stringy_base_ty_path(base);
                quote! {
                    (#binding).as_ref().map(|#v| {
                        <#ty_path as ::core::convert::AsRef<str>>::as_ref(#v)
                    })
                }
            }
        }
        StringyExprKind::CollapsedOption => {
            let v = idents::leaf_value();
            if is_string {
                quote! { (#binding).map(|#v| #v.as_str()) }
            } else {
                let ty_path = stringy_base_ty_path(base);
                quote! {
                    (#binding).map(|#v| {
                        <#ty_path as ::core::convert::AsRef<str>>::as_ref(#v)
                    })
                }
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

/// Build a fallible generated expression for the current flat-ref index.
/// This feeds `IdxCa::take` positions, so truncation under `idx-u32` must
/// surface as a Polars error instead of silently wrapping.
pub(super) fn idx_size_len_expr(flat: &syn::Ident, pp: &TokenStream) -> TokenStream {
    quote! {
        <#pp::IdxSize as ::core::convert::TryFrom<usize>>::try_from(#flat.len())
            .map_err(|_| #pp::polars_err!(
                ComputeError:
                "df-derive: nested inner-option position {} exceeds IdxSize range",
                #flat.len(),
            ))?
    }
}

/// Build a fallible generated expression for list offsets. Polars large-list
/// offsets are `i64`, so impossible `usize` lengths should surface as a
/// Polars error instead of silently wrapping through `as`.
pub(super) fn list_offset_i64_expr(offset: &TokenStream, pp: &TokenStream) -> TokenStream {
    quote! {
        <i64 as ::core::convert::TryFrom<usize>>::try_from(#offset)
            .map_err(|_| #pp::polars_err!(
                ComputeError:
                "df-derive: list offset {} exceeds i64 range",
                #offset,
            ))?
    }
}

/// Collapse `n` `Option` layers into one `Option<&Inner>`.
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

pub(super) struct ChainRef {
    pub expr: TokenStream,
    pub has_option: bool,
}

fn deref_ref_expr(base_ref: &TokenStream, smart_ptrs: usize) -> TokenStream {
    if smart_ptrs == 0 {
        return base_ref.clone();
    }
    let mut out = quote! { *(#base_ref) };
    for _ in 0..smart_ptrs {
        out = quote! { *(#out) };
    }
    quote! { (&(#out)) }
}

fn apply_pending_smart_ptrs(expr: TokenStream, has_option: bool, smart_ptrs: usize) -> TokenStream {
    if smart_ptrs == 0 {
        return expr;
    }
    if has_option {
        let param = idents::collapse_option_param();
        let derefed = deref_ref_expr(&quote! { #param }, smart_ptrs);
        quote! { (#expr).map(|#param| #derefed) }
    } else {
        deref_ref_expr(&expr, smart_ptrs)
    }
}

/// Resolve a transparent access chain at one wrapper boundary.
pub(super) fn access_chain_to_ref(base: &TokenStream, chain: &AccessChain) -> ChainRef {
    if chain.is_empty() {
        return ChainRef {
            expr: base.clone(),
            has_option: false,
        };
    }
    if chain.is_only_options() {
        let option_layers = chain.option_layers();
        return ChainRef {
            expr: if option_layers == 1 {
                base.clone()
            } else {
                collapse_options_to_ref(base, option_layers)
            },
            has_option: option_layers > 0,
        };
    }

    let mut expr = base.clone();
    let mut has_option = false;
    let mut pending_smart_ptrs = 0usize;

    for step in &chain.steps {
        match step {
            AccessStep::SmartPtr => {
                pending_smart_ptrs += 1;
            }
            AccessStep::Option => {
                expr = apply_pending_smart_ptrs(expr, has_option, pending_smart_ptrs);
                pending_smart_ptrs = 0;
                let param = idents::collapse_option_param();
                expr = if has_option {
                    quote! { (#expr).and_then(|#param| (#param).as_ref()) }
                } else {
                    has_option = true;
                    quote! { (#expr).as_ref() }
                };
            }
        }
    }

    expr = apply_pending_smart_ptrs(expr, has_option, pending_smart_ptrs);
    ChainRef { expr, has_option }
}

/// Resolve an optional access chain into a mappable `Option<&T>`.
pub(super) fn access_chain_to_option_ref(base: &TokenStream, chain: &AccessChain) -> TokenStream {
    debug_assert!(chain.option_layers() > 0);
    if chain.is_only_options() {
        collapse_options_to_ref(base, chain.option_layers())
    } else {
        access_chain_to_ref(base, chain).expr
    }
}

pub enum Encoder {
    Leaf {
        decls: Vec<TokenStream>,
        push: TokenStream,
        series: TokenStream,
    },
    Multi {
        columnar: TokenStream,
    },
}

pub struct BaseCtx<'a> {
    pub access: &'a TokenStream,
    pub idx: usize,
    pub name: &'a str,
}

pub struct LeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub decimal128_encode_trait: &'a syn::Path,
    pub paths: &'a ExternalPaths,
}

pub fn build_encoder(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    ctx: &LeafCtx<'_>,
) -> Encoder {
    build_encoder_with_option_receiver(leaf, wrapper, ctx, None)
}

pub(super) fn build_encoder_with_option_receiver(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    ctx: &LeafCtx<'_>,
    option_some_receiver: Option<crate::codegen::type_registry::PrimitiveExprReceiver>,
) -> Encoder {
    match wrapper {
        WrapperShape::Leaf(LeafShape::Bare) => {
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
        WrapperShape::Leaf(LeafShape::Optional {
            option_layers,
            access,
        }) if option_layers.get() == 1 && access.is_single_plain_option() => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(
                leaf,
                ctx,
                leaf::LeafArmKind::Option {
                    some_receiver: option_some_receiver
                        .unwrap_or(crate::codegen::type_registry::PrimitiveExprReceiver::Ref),
                },
            );
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        // Polars folds every nested None to one validity bit.
        WrapperShape::Leaf(LeafShape::Optional {
            option_layers,
            access,
        }) => option::wrap_option_access_chain_primitive(leaf, ctx, access, option_layers.get()),
        WrapperShape::Vec(vec_layers) => vec::try_build_vec_encoder(leaf, ctx, vec_layers),
    }
}
