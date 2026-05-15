use syn::{AngleBracketedGenericArguments, GenericArgument, PathArguments, Type, TypePath};

use super::known_types::{is_bare_str_type, is_u8_type};
use super::path_match::{path_is_exact_with_leaf_args, wrapper_path_matches};
use super::{AnalyzedBase, RawWrapper};

pub(super) struct PeeledType<'a> {
    pub(super) wrappers: Vec<RawWrapper>,
    pub(super) current_type: &'a Type,
    pub(super) outer_smart_ptr_depth: usize,
}

fn record_smart_ptr_layer(outer: &mut usize, wrappers: &mut Vec<RawWrapper>) {
    if wrappers.is_empty() {
        *outer += 1;
    } else {
        wrappers.push(RawWrapper::SmartPtr);
    }
}

fn peel_option(ty: &Type) -> Option<&Type> {
    extract_inner_type(ty, "Option", &["std", "option", "Option"])
        .or_else(|| extract_inner_type(ty, "Option", &["core", "option", "Option"]))
}

fn peel_vec(ty: &Type) -> Option<&Type> {
    extract_inner_type(ty, "Vec", &["std", "vec", "Vec"])
        .or_else(|| extract_inner_type(ty, "Vec", &["alloc", "vec", "Vec"]))
}

fn peel_smart_ptr(ty: &Type) -> Option<&Type> {
    extract_inner_type(ty, "Box", &["std", "boxed", "Box"])
        .or_else(|| extract_inner_type(ty, "Box", &["alloc", "boxed", "Box"]))
        .or_else(|| extract_inner_type(ty, "Rc", &["std", "rc", "Rc"]))
        .or_else(|| extract_inner_type(ty, "Rc", &["alloc", "rc", "Rc"]))
        .or_else(|| extract_inner_type(ty, "Arc", &["std", "sync", "Arc"]))
        .or_else(|| extract_inner_type(ty, "Arc", &["alloc", "sync", "Arc"]))
}

fn peel_reference(ty: &Type) -> Result<Option<&Type>, syn::Error> {
    let Type::Reference(reference) = ty else {
        return Ok(None);
    };

    if reference.mutability.is_some() {
        return Err(syn::Error::new_spanned(
            reference,
            "df-derive does not support `&mut T` fields; use `&T`, an owned value, \
             or mark the field `#[df_derive(skip)]`",
        ));
    }
    if borrowed_reference_base(reference).is_some() {
        return Ok(None);
    }

    Ok(Some(reference.elem.as_ref()))
}

pub(super) fn peel_type_wrappers(ty: &Type) -> Result<PeeledType<'_>, syn::Error> {
    let mut wrappers: Vec<RawWrapper> = Vec::new();
    let mut outer_smart_ptr_depth: usize = 0;
    let mut current_type = ty;

    loop {
        if let Some(inner_ty) = peel_option(current_type) {
            wrappers.push(RawWrapper::Option);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = peel_vec(current_type) {
            wrappers.push(RawWrapper::Vec);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = peel_smart_ptr(current_type) {
            record_smart_ptr_layer(&mut outer_smart_ptr_depth, &mut wrappers);
            current_type = inner_ty;
            continue;
        }
        if let Some(action) = peel_cow(current_type) {
            match action {
                CowPeel::Rebind(inner) => {
                    record_smart_ptr_layer(&mut outer_smart_ptr_depth, &mut wrappers);
                    current_type = inner;
                    continue;
                }
                CowPeel::KeepAsSemanticBase => break,
            }
        }
        if let Some(inner_ty) = peel_reference(current_type)? {
            record_smart_ptr_layer(&mut outer_smart_ptr_depth, &mut wrappers);
            current_type = inner_ty;
            continue;
        }
        break;
    }

    Ok(PeeledType {
        wrappers,
        current_type,
        outer_smart_ptr_depth,
    })
}

fn extract_inner_type<'a>(ty: &'a Type, wrapper: &str, qualified: &[&str]) -> Option<&'a Type> {
    if let Type::Path(type_path) = ty
        && wrapper_path_matches(type_path, wrapper, qualified)
        && let Some(segment) = type_path.path.segments.last()
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(inner_ty) = single_type_arg(args)
    {
        return Some(inner_ty);
    }
    None
}

fn single_type_arg(args: &AngleBracketedGenericArguments) -> Option<&Type> {
    let mut args = args.args.iter();
    let first = args.next()?;
    if args.next().is_some() {
        return None;
    }
    match first {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    }
}

enum CowPeel<'a> {
    Rebind(&'a Type),
    KeepAsSemanticBase,
}

fn peel_cow(ty: &Type) -> Option<CowPeel<'_>> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    if !is_cow_path(type_path) {
        return None;
    }
    let inner_ty = cow_inner_type(type_path)?;
    if is_bare_str_type(inner_ty) {
        return Some(CowPeel::KeepAsSemanticBase);
    }
    if matches!(inner_ty, Type::Slice(_)) {
        return Some(CowPeel::KeepAsSemanticBase);
    }
    Some(CowPeel::Rebind(inner_ty))
}

fn is_cow_path(type_path: &TypePath) -> bool {
    path_is_exact_with_leaf_args(type_path, &["Cow"])
        || path_is_exact_with_leaf_args(type_path, &["std", "borrow", "Cow"])
        || path_is_exact_with_leaf_args(type_path, &["alloc", "borrow", "Cow"])
}

fn cow_inner_type(type_path: &TypePath) -> Option<&Type> {
    if !is_cow_path(type_path) {
        return None;
    }
    let segment = type_path.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}

pub(super) fn borrowed_reference_base(reference: &syn::TypeReference) -> Option<AnalyzedBase> {
    let inner_ty = reference.elem.as_ref();
    if is_bare_str_type(inner_ty) {
        return Some(AnalyzedBase::BorrowedStr);
    }
    if let Type::Slice(slice) = inner_ty {
        if is_u8_type(&slice.elem) {
            Some(AnalyzedBase::BorrowedBytes)
        } else {
            Some(AnalyzedBase::BorrowedSlice)
        }
    } else {
        None
    }
}

pub(super) fn analyze_cow_base(type_path: &TypePath) -> Option<AnalyzedBase> {
    let inner_ty = cow_inner_type(type_path)?;
    if is_bare_str_type(inner_ty) {
        return Some(AnalyzedBase::CowStr);
    }
    if let Type::Slice(slice) = inner_ty {
        if is_u8_type(&slice.elem) {
            Some(AnalyzedBase::CowBytes)
        } else {
            Some(AnalyzedBase::CowSlice)
        }
    } else {
        None
    }
}
