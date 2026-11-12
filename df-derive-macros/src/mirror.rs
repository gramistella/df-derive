//! Builder types for fields that carry an encoding attribute.
//!
//! An encoding such as `as_str` applies to the innermost value of a field.
//! The wrapper layers around it are read from the type as written, and the
//! builder type mirrors them: `Option<Vec<Venue>>` with `as_str` becomes
//! `OptionBuilder<ListBuilder<StrBuilder>>`. The innermost value of a map is
//! its value; its keys keep the encoding of their own type. Types the derive
//! does not recognise as a wrapper are treated as the leaf, so a mismatch
//! surfaces as an ordinary trait-bound error on the field.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericArgument, PathArguments, Type};

use crate::attrs::{Encoding, TimeUnit};

enum Wrapper<'a> {
    Option(&'a Type),
    Vec(&'a Type),
    Slice(&'a Type),
    Pointer(&'a Type),
    Map { key: &'a Type, value: &'a Type },
}

fn peel(ty: &Type) -> &Type {
    match ty {
        Type::Paren(inner) => peel(&inner.elem),
        Type::Group(inner) => peel(&inner.elem),
        other => other,
    }
}

/// The type arguments of `Name<..>`, spelled bare or through one of its
/// standard paths.
fn std_arguments<'a>(path: &'a syn::Path, name: &str, modules: &[&[&str]]) -> Vec<&'a Type> {
    let segments: Vec<_> = path.segments.iter().collect();
    let Some((last, prefix)) = segments.split_last() else {
        return Vec::new();
    };
    let prefix_matches = prefix.is_empty()
        || modules.iter().any(|module| {
            module.len() == prefix.len()
                && module.iter().zip(prefix).all(|(expected, segment)| {
                    segment.ident == expected && segment.arguments.is_none()
                })
        });
    if last.ident != name || !prefix_matches {
        return Vec::new();
    }
    let PathArguments::AngleBracketed(arguments) = &last.arguments else {
        return Vec::new();
    };
    arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect()
}

/// Matches a standard type with exactly one type argument.
fn std_generic<'a>(path: &'a syn::Path, name: &str, modules: &[&[&str]]) -> Option<&'a Type> {
    match std_arguments(path, name, modules).as_slice() {
        [only] => Some(only),
        _ => None,
    }
}

/// Matches `HashMap<K, V>`, with or without a hasher, and `BTreeMap<K, V>`.
fn std_map(path: &syn::Path) -> Option<Wrapper<'_>> {
    let hash = std_arguments(
        path,
        "HashMap",
        &[&["std", "collections"], &["std", "collections", "hash_map"]],
    );
    let tree = std_arguments(
        path,
        "BTreeMap",
        &[
            &["std", "collections"],
            &["std", "collections", "btree_map"],
            &["alloc", "collections"],
            &["alloc", "collections", "btree_map"],
        ],
    );
    match (hash.as_slice(), tree.as_slice()) {
        ([key, value] | [key, value, _], _) | (_, [key, value]) => {
            Some(Wrapper::Map { key, value })
        }
        _ => None,
    }
}

fn wrapper(ty: &Type) -> Option<Wrapper<'_>> {
    match ty {
        Type::Reference(reference) if reference.mutability.is_none() => {
            Some(Wrapper::Pointer(&reference.elem))
        }
        Type::Slice(slice) => Some(Wrapper::Slice(&slice.elem)),
        Type::Path(path) if path.qself.is_none() => {
            let path = &path.path;
            if let Some(inner) =
                std_generic(path, "Option", &[&["std", "option"], &["core", "option"]])
            {
                return Some(Wrapper::Option(inner));
            }
            if let Some(inner) = std_generic(path, "Vec", &[&["std", "vec"], &["alloc", "vec"]]) {
                return Some(Wrapper::Vec(inner));
            }
            if let Some(map) = std_map(path) {
                return Some(map);
            }
            let pointers: [(&str, &[&[&str]]); 4] = [
                ("Box", &[&["std", "boxed"], &["alloc", "boxed"]]),
                ("Rc", &[&["std", "rc"], &["alloc", "rc"]]),
                ("Arc", &[&["std", "sync"], &["alloc", "sync"]]),
                ("Cow", &[&["std", "borrow"], &["alloc", "borrow"]]),
            ];
            pointers
                .iter()
                .find_map(|(name, modules)| std_generic(path, name, modules))
                .map(Wrapper::Pointer)
        }
        _ => None,
    }
}

fn is_u8(ty: &Type) -> bool {
    matches!(peel(ty), Type::Path(path) if path.qself.is_none() && path.path.is_ident("u8"))
}

fn leaf(encoding: &Encoding, builders: &TokenStream) -> TokenStream {
    match encoding {
        Encoding::AsStr => quote! { #builders::StrBuilder },
        Encoding::AsString => quote! { #builders::DisplayBuilder },
        Encoding::AsBinary => quote! { #builders::BinaryBuilder },
        Encoding::Decimal { precision, scale } => {
            let precision = usize::from(*precision);
            let scale = usize::from(*scale);
            quote! { #builders::DecimalBuilder<#precision, #scale> }
        }
        Encoding::TimeUnit(unit) => {
            let unit = match unit {
                TimeUnit::Ms => quote! { Ms },
                TimeUnit::Us => quote! { Us },
                TimeUnit::Ns => quote! { Ns },
            };
            quote! { #builders::TemporalBuilder<#builders::#unit> }
        }
        Encoding::With(builder) => quote! { #builder },
    }
}

/// Returns the builder type for `ty` under `encoding`.
///
/// `root` is the runtime's `dataframe` module and `builders` its `builders`
/// module.
pub fn builder(
    ty: &Type,
    encoding: &Encoding,
    root: &TokenStream,
    builders: &TokenStream,
) -> TokenStream {
    let ty = peel(ty);
    // A byte sequence is the value `as_binary` encodes, not a list around it.
    let bytes = matches!(encoding, Encoding::AsBinary);
    match wrapper(ty) {
        Some(Wrapper::Option(inner)) => {
            let inner = builder(inner, encoding, root, builders);
            quote! { #builders::OptionBuilder<#inner> }
        }
        Some(Wrapper::Vec(inner) | Wrapper::Slice(inner)) if !(bytes && is_u8(inner)) => {
            let inner = builder(inner, encoding, root, builders);
            quote! { #builders::ListBuilder<#inner> }
        }
        Some(Wrapper::Pointer(inner)) => {
            let inner = builder(inner, encoding, root, builders);
            quote! { #builders::PtrBuilder<#inner> }
        }
        Some(Wrapper::Map { key, value }) => {
            let value = builder(value, encoding, root, builders);
            quote! { #builders::MapBuilder<<#key as #root::Columnar>::Builder, #value> }
        }
        _ => leaf(encoding, builders),
    }
}

/// Whether the innermost value of `ty` is a tuple.
pub fn leaf_is_tuple(ty: &Type) -> bool {
    let ty = peel(ty);
    match wrapper(ty) {
        Some(
            Wrapper::Option(inner)
            | Wrapper::Vec(inner)
            | Wrapper::Slice(inner)
            | Wrapper::Pointer(inner)
            | Wrapper::Map { value: inner, .. },
        ) => leaf_is_tuple(inner),
        None => matches!(ty, Type::Tuple(_)),
    }
}

/// Whether `ty` contains the deriving type itself, written as `Self`, as its
/// bare name, or as `self::Name` / `crate::Name`.
pub fn mentions_self(ty: &Type, name: &syn::Ident) -> bool {
    let any = |types: &mut dyn Iterator<Item = &Type>| {
        let mut found = false;
        for ty in types {
            found |= mentions_self(ty, name);
        }
        found
    };
    match ty {
        Type::Paren(inner) => mentions_self(&inner.elem, name),
        Type::Group(inner) => mentions_self(&inner.elem, name),
        Type::Reference(inner) => mentions_self(&inner.elem, name),
        Type::Slice(inner) => mentions_self(&inner.elem, name),
        Type::Array(inner) => mentions_self(&inner.elem, name),
        Type::Ptr(inner) => mentions_self(&inner.elem, name),
        Type::Tuple(tuple) => any(&mut tuple.elems.iter()),
        Type::Path(path) if path.qself.is_none() => {
            let segments: Vec<_> = path.path.segments.iter().collect();
            let is_self = match segments.as_slice() {
                [only] => only.ident == "Self" || only.ident == *name,
                [scope, last] => {
                    (scope.ident == "self" || scope.ident == "crate") && last.ident == *name
                }
                _ => false,
            };
            is_self
                || segments.iter().any(|segment| match &segment.arguments {
                    PathArguments::AngleBracketed(arguments) => any(&mut arguments
                        .args
                        .iter()
                        .filter_map(|argument| match argument {
                            GenericArgument::Type(ty) => Some(ty),
                            _ => None,
                        })),
                    _ => false,
                })
        }
        _ => false,
    }
}
