use proc_macro2::TokenStream;
use quote::quote;
use syn::PathArguments;

use crate::ir::StringyBase;

use super::idents;

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

pub(in crate::codegen) fn stringy_base_ty_path(base: &StringyBase) -> TokenStream {
    match base {
        StringyBase::String => quote! { ::std::string::String },
        StringyBase::BorrowedStr => quote! { &'_ str },
        StringyBase::CowStr => quote! { ::std::borrow::Cow<'_, str> },
        StringyBase::Struct(ty) => struct_type_tokens(ty),
        StringyBase::Generic(ident) => quote! { #ident },
    }
}

#[derive(Clone, Copy)]
pub(in crate::codegen) enum StringyExprKind {
    Bare,
    /// Uses closures so smart-pointer deref coercions fire below `Option`.
    OptionDeref,
    CollapsedOption,
    MbvaValue,
}

pub(in crate::codegen) fn stringy_value_expr(
    base: &StringyBase,
    binding: &TokenStream,
    kind: StringyExprKind,
) -> TokenStream {
    if matches!(base, StringyBase::BorrowedStr | StringyBase::CowStr) {
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
