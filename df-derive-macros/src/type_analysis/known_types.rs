use crate::ir::NumericKind;
use syn::{GenericArgument, PathArguments, Type, TypePath};

use super::AnalyzedBase;
use super::path_match::{PathView, path_is_exact_no_args, path_is_exact_with_leaf_args};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KnownBase {
    Numeric(NumericKind),
    Bool,
    String,
    Decimal,
    StdDuration,
    ChronoDuration,
    DateTimeTz,
    NaiveDate,
    NaiveTime,
    NaiveDateTime,
}

impl KnownBase {
    pub(super) const fn into_analyzed_base(self) -> AnalyzedBase {
        match self {
            Self::Numeric(kind) => AnalyzedBase::Numeric(kind),
            Self::Bool => AnalyzedBase::Bool,
            Self::String => AnalyzedBase::String,
            Self::Decimal => AnalyzedBase::Decimal,
            Self::StdDuration => AnalyzedBase::StdDuration,
            Self::ChronoDuration => AnalyzedBase::ChronoDuration,
            Self::DateTimeTz => AnalyzedBase::DateTimeTz,
            Self::NaiveDate => AnalyzedBase::NaiveDate,
            Self::NaiveTime => AnalyzedBase::NaiveTime,
            Self::NaiveDateTime => AnalyzedBase::NaiveDateTime,
        }
    }
}

pub(super) fn classify_known_base(type_path: &TypePath) -> Option<KnownBase> {
    if is_std_duration(type_path) {
        return Some(KnownBase::StdDuration);
    }
    if is_chrono_duration(type_path) {
        return Some(KnownBase::ChronoDuration);
    }
    if is_chrono_datetime(type_path) {
        return Some(KnownBase::DateTimeTz);
    }
    if let Some(kind) = bare_numeric_kind(type_path).or_else(|| nonzero_numeric_kind(type_path)) {
        return Some(KnownBase::Numeric(kind));
    }
    if path_is_exact_no_args(type_path, &["bool"]) {
        return Some(KnownBase::Bool);
    }
    if is_string_type(type_path) {
        return Some(KnownBase::String);
    }
    if is_decimal_type(type_path) {
        return Some(KnownBase::Decimal);
    }
    if is_chrono_no_args_type(type_path, "NaiveDate") {
        return Some(KnownBase::NaiveDate);
    }
    if is_chrono_no_args_type(type_path, "NaiveTime") {
        return Some(KnownBase::NaiveTime);
    }
    if is_chrono_no_args_type(type_path, "NaiveDateTime") {
        return Some(KnownBase::NaiveDateTime);
    }
    None
}

fn bare_numeric_kind(type_path: &TypePath) -> Option<NumericKind> {
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    match segment.ident.to_string().as_str() {
        "f64" => Some(NumericKind::F64),
        "f32" => Some(NumericKind::F32),
        "i8" => Some(NumericKind::I8),
        "u8" => Some(NumericKind::U8),
        "i16" => Some(NumericKind::I16),
        "u16" => Some(NumericKind::U16),
        "i64" => Some(NumericKind::I64),
        "i128" => Some(NumericKind::I128),
        "isize" => Some(NumericKind::ISize),
        "u64" => Some(NumericKind::U64),
        "u128" => Some(NumericKind::U128),
        "usize" => Some(NumericKind::USize),
        "u32" => Some(NumericKind::U32),
        "i32" => Some(NumericKind::I32),
        _ => None,
    }
}

fn nonzero_numeric_kind(type_path: &TypePath) -> Option<NumericKind> {
    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    let kind = match segment.ident.to_string().as_str() {
        "NonZeroI8" => NumericKind::NonZeroI8,
        "NonZeroI16" => NumericKind::NonZeroI16,
        "NonZeroI32" => NumericKind::NonZeroI32,
        "NonZeroI64" => NumericKind::NonZeroI64,
        "NonZeroI128" => NumericKind::NonZeroI128,
        "NonZeroIsize" => NumericKind::NonZeroISize,
        "NonZeroU8" => NumericKind::NonZeroU8,
        "NonZeroU16" => NumericKind::NonZeroU16,
        "NonZeroU32" => NumericKind::NonZeroU32,
        "NonZeroU64" => NumericKind::NonZeroU64,
        "NonZeroU128" => NumericKind::NonZeroU128,
        "NonZeroUsize" => NumericKind::NonZeroUSize,
        _ => return None,
    };
    let leaf = segment.ident.to_string();
    let leaf = leaf.as_str();
    if path_is_exact_no_args(type_path, &[leaf])
        || path_is_exact_no_args(type_path, &["std", "num", leaf])
        || path_is_exact_no_args(type_path, &["core", "num", leaf])
    {
        Some(kind)
    } else {
        None
    }
}

fn is_string_type(type_path: &TypePath) -> bool {
    let Some(path) = PathView::from_type_path(type_path) else {
        return false;
    };

    path.exact_no_args(&["String"])
        || path.exact_no_args(&["std", "string", "String"])
        || path.exact_no_args(&["alloc", "string", "String"])
}

fn is_decimal_type(type_path: &TypePath) -> bool {
    let Some(path) = PathView::from_type_path(type_path) else {
        return false;
    };

    path.exact_no_args(&["Decimal"]) || path.exact_no_args(&["rust_decimal", "Decimal"])
}

fn is_chrono_no_args_type(type_path: &TypePath, leaf: &str) -> bool {
    let Some(path) = PathView::from_type_path(type_path) else {
        return false;
    };

    path.exact_no_args(&[leaf]) || path.exact_no_args(&["chrono", leaf])
}

fn is_std_duration(type_path: &TypePath) -> bool {
    let Some(path) = PathView::from_type_path(type_path) else {
        return false;
    };

    path.exact_no_args(&["std", "time", "Duration"])
        || path.exact_no_args(&["core", "time", "Duration"])
}

fn is_chrono_duration(type_path: &TypePath) -> bool {
    let Some(path) = PathView::from_type_path(type_path) else {
        return false;
    };
    let Some(segment) = path.leaf() else {
        return false;
    };

    if !matches!(segment.arguments, PathArguments::None) {
        return false;
    }

    if segment.ident == "Duration" {
        path.exact_no_args(&["Duration"]) || path.exact_no_args(&["chrono", "Duration"])
    } else if segment.ident == "TimeDelta" {
        path.exact_no_args(&["TimeDelta"]) || path.exact_no_args(&["chrono", "TimeDelta"])
    } else {
        false
    }
}

fn is_chrono_datetime(type_path: &TypePath) -> bool {
    if !path_is_exact_with_leaf_args(type_path, &["DateTime"])
        && !path_is_exact_with_leaf_args(type_path, &["chrono", "DateTime"])
    {
        return false;
    }
    let Some(segment) = type_path.path.segments.last() else {
        return false;
    };
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return false;
    };
    args.args
        .iter()
        .any(|arg| matches!(arg, GenericArgument::Type(_)))
}

pub(super) fn is_bare_str_type(ty: &Type) -> bool {
    if let Type::Path(inner_path) = ty
        && inner_path.qself.is_none()
        && inner_path.path.segments.len() == 1
        && let Some(seg) = inner_path.path.segments.last()
    {
        return seg.ident == "str" && matches!(seg.arguments, PathArguments::None);
    }
    false
}

pub(super) fn is_u8_type(ty: &Type) -> bool {
    if let Type::Path(inner_path) = ty
        && inner_path.qself.is_none()
        && inner_path.path.segments.len() == 1
        && let Some(seg) = inner_path.path.segments.last()
    {
        return seg.ident == "u8" && matches!(seg.arguments, PathArguments::None);
    }
    false
}
