use crate::ir::{BaseType, DateTimeUnit, PrimitiveTransform, Wrapper};
use syn::{GenericArgument, Ident, PathArguments, Type};

/// Default `Datetime` precision for `chrono::DateTime<Utc>` fields without an
/// explicit `time_unit` override. Matches the historical default this crate
/// shipped with.
pub const DEFAULT_DATETIME_UNIT: DateTimeUnit = DateTimeUnit::Milliseconds;
/// Default `Decimal(precision, scale)` for `rust_decimal::Decimal` fields
/// without an explicit `decimal(...)` override.
pub const DEFAULT_DECIMAL_PRECISION: u8 = 38;
/// Default scale paired with `DEFAULT_DECIMAL_PRECISION`.
pub const DEFAULT_DECIMAL_SCALE: u8 = 10;

#[derive(Clone)]
pub struct AnalyzedType {
    pub base: BaseType,
    pub wrappers: Vec<Wrapper>,
    pub transform: Option<PrimitiveTransform>,
}

pub fn analyze_type(
    ty: &Type,
    as_string: bool,
    as_str: bool,
    generic_params: &[Ident],
) -> Result<AnalyzedType, syn::Error> {
    let mut wrappers: Vec<Wrapper> = Vec::new();
    let mut current_type = ty;

    // Loop to peel off wrappers in any order
    loop {
        if let Some(inner_ty) = extract_inner_type(current_type, "Option") {
            wrappers.push(Wrapper::Option);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = extract_inner_type(current_type, "Vec") {
            wrappers.push(Wrapper::Vec);
            current_type = inner_ty;
            continue;
        }
        // No more wrappers found, break the loop
        break;
    }

    // Before resolving the base type, reject a small allow-list of common
    // wrapper / collection types with an actionable hint. These all parse
    // fine as a `Type::Path` and would otherwise either fall through to the
    // generic "Unsupported field type" error or — worse — be silently
    // routed through the `Struct` arm and explode at codegen time.
    if let Type::Path(type_path) = current_type
        && let Some(segment) = type_path.path.segments.last()
    {
        let hint: Option<&'static str> = match segment.ident.to_string().as_str() {
            "HashMap" => Some(
                "df-derive does not support `HashMap` fields. Convert to \
                 `Vec<(K, V)>` or pre-flatten into named columns before assignment.",
            ),
            "BTreeMap" => Some(
                "df-derive does not support `BTreeMap` fields. Convert to \
                 `Vec<(K, V)>` or pre-flatten into named columns before assignment.",
            ),
            "HashSet" => Some(
                "df-derive does not support `HashSet` fields. Convert to \
                 `Vec<T>` (order will be set-defined, not insertion-defined).",
            ),
            "Box" => Some(
                "df-derive does not support `Box` fields. Use the inner type \
                 directly; df-derive copies values during conversion, so `Box` \
                 only adds heap indirection without changing the column shape.",
            ),
            "Rc" => Some(
                "df-derive does not support `Rc` fields. Use the inner type \
                 directly; df-derive copies values during conversion, so the \
                 reference count cannot be preserved into a Polars column.",
            ),
            "Arc" => Some(
                "df-derive does not support `Arc` fields. Use the inner type \
                 directly; df-derive copies values during conversion, so the \
                 reference count cannot be preserved into a Polars column.",
            ),
            "Cow" => Some(
                "df-derive does not support `Cow` fields. Use the owned type \
                 directly; the conversion always materializes owned values.",
            ),
            _ => None,
        };
        if let Some(message) = hint {
            return Err(syn::Error::new_spanned(current_type, message));
        }
    }

    let base = analyze_base_type(current_type, generic_params)
        .ok_or_else(|| syn::Error::new_spanned(current_type, "Unsupported field type"))?;

    // Determine abstract transform; `as_str` and `as_string` attributes override
    // any base-type-driven default. The two attributes are mutually exclusive
    // (enforced in the parser).
    let transform = if as_str {
        Some(PrimitiveTransform::AsStr)
    } else if as_string {
        Some(PrimitiveTransform::ToString)
    } else {
        match &base {
            BaseType::DateTimeUtc => Some(PrimitiveTransform::DateTimeToInt(DEFAULT_DATETIME_UNIT)),
            BaseType::Decimal => Some(PrimitiveTransform::DecimalToInt128 {
                precision: DEFAULT_DECIMAL_PRECISION,
                scale: DEFAULT_DECIMAL_SCALE,
            }),
            _ => None,
        }
    };

    Ok(AnalyzedType {
        base,
        wrappers,
        transform,
    })
}

fn analyze_base_type(ty: &Type, generic_params: &[Ident]) -> Option<BaseType> {
    if is_datetime_utc(ty) {
        return Some(BaseType::DateTimeUtc);
    }
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        let type_ident = &segment.ident;
        let has_args = !matches!(segment.arguments, PathArguments::None);
        let is_single_segment = type_path.qself.is_none() && type_path.path.segments.len() == 1;
        let base_type = match type_ident.to_string().as_str() {
            "String" => BaseType::String,
            "f64" => BaseType::F64,
            "f32" => BaseType::F32,
            "i8" => BaseType::I8,
            "u8" => BaseType::U8,
            "i16" => BaseType::I16,
            "u16" => BaseType::U16,
            "i64" => BaseType::I64,
            "isize" => BaseType::ISize,
            "u64" => BaseType::U64,
            "usize" => BaseType::USize,
            "u32" => BaseType::U32,
            "i32" => BaseType::I32,
            "bool" => BaseType::Bool,
            "Decimal" => BaseType::Decimal,
            _ => {
                if is_single_segment && !has_args && generic_params.iter().any(|p| p == type_ident)
                {
                    BaseType::Generic(type_ident.clone())
                } else {
                    let args = match &segment.arguments {
                        PathArguments::AngleBracketed(ab) => Some(ab.clone()),
                        _ => None,
                    };
                    BaseType::Struct(type_ident.clone(), args)
                }
            }
        };
        return Some(base_type);
    }
    None
}

fn extract_inner_type<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == wrapper
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
    {
        return Some(inner_ty);
    }
    None
}

/// Detect a `chrono::DateTime<Utc>` field by ident only.
///
/// The match looks at the last segment of the outer path (`DateTime`) and the
/// last segment of the first generic argument's path (`Utc`). Anything that
/// happens to share those idents — e.g. `some_other_crate::DateTime<other::Utc>`
/// — would be a false positive and routed through the chrono encoder.
///
/// This leniency is intentional: user crates frequently re-export
/// `chrono::DateTime<chrono::Utc>` under their own paths (type aliases, prelude
/// modules, glob re-exports), and tightening the match to a specific path
/// prefix would break those uses without a robust way to recover the original
/// definition from a `syn::Type` alone. The proc-macro happily generates code
/// that calls chrono's `timestamp_*` methods; if the type isn't actually
/// chrono's, the user gets a compile error at the call site, which is the
/// correct failure mode.
fn is_datetime_utc(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "DateTime"
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(Type::Path(inner))) = args.args.first()
        && let Some(inner_seg) = inner.path.segments.last()
    {
        return inner_seg.ident == "Utc";
    }
    false
}
