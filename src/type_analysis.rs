use crate::ir::{DateTimeUnit, NumericKind};
use syn::{GenericArgument, Ident, PathArguments, Type, TypePath};

/// Default `Datetime` precision for `chrono::DateTime<Utc>` fields without an
/// explicit `time_unit` override. Matches the historical default this crate
/// shipped with.
pub const DEFAULT_DATETIME_UNIT: DateTimeUnit = DateTimeUnit::Milliseconds;
/// Default `Duration` precision for `std::time::Duration` and
/// `chrono::Duration` (`chrono::TimeDelta`) fields without an explicit
/// `time_unit` override. Nanoseconds is the most-information-preserving
/// choice and matches `polars-arrow`'s default `Duration` representation.
pub const DEFAULT_DURATION_UNIT: DateTimeUnit = DateTimeUnit::Nanoseconds;
/// Default `Decimal(precision, scale)` for `rust_decimal::Decimal` fields
/// without an explicit `decimal(...)` override.
pub const DEFAULT_DECIMAL_PRECISION: u8 = 38;
/// Default scale paired with `DEFAULT_DECIMAL_PRECISION`.
pub const DEFAULT_DECIMAL_SCALE: u8 = 10;

/// Raw wrapper position before normalization. The parser collapses these
/// into a `WrapperShape` (with consecutive `Option`s folded per Polars's
/// single-validity-bit-per-level representation).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RawWrapper {
    Option,
    Vec,
}

/// Analyzed base type the parser uses internally before folding through the
/// `(override, base)` legality matrix into a `LeafSpec`. Distinct from the
/// IR's `LeafSpec` because this layer doesn't yet carry override-dependent
/// information (e.g. `Decimal { precision, scale }`, `DateTime(unit)`,
/// stringy classification) — that fusion happens in the parser when the
/// override is consulted.
#[derive(Clone)]
pub enum AnalyzedBase {
    Numeric(NumericKind),
    String,
    Bool,
    DateTimeUtc,
    /// `chrono::NaiveDate` — last-segment `NaiveDate` with no generic args
    /// matches, mirroring `is_datetime_utc`'s leniency. The encoder emits
    /// chrono calls; a same-name false positive surfaces as a compile
    /// error at the call site.
    NaiveDate,
    /// `chrono::NaiveTime` — last-segment `NaiveTime` with no generic args
    /// matches, same leniency as [`Self::NaiveDate`].
    NaiveTime,
    /// `std::time::Duration` (or `core::time::Duration`). Detected by path
    /// segment matching to disambiguate from `chrono::Duration`; bare
    /// `Duration` is rejected as ambiguous in [`analyze_type`].
    StdDuration,
    /// `chrono::Duration` (alias for `chrono::TimeDelta`). Detected by path
    /// segment matching. Codegen uses the user's declared field-type tokens
    /// directly so type inference resolves the alias correctly.
    ChronoDuration,
    Decimal,
    /// Concrete user-defined struct, with optional angle-bracketed generic
    /// arguments at the field's use site (e.g. `<M>` in `Vec<Foo<M>>`).
    Struct(Ident, Option<syn::AngleBracketedGenericArguments>),
    /// Generic type parameter declared on the enclosing struct.
    Generic(Ident),
}

#[derive(Clone)]
pub struct AnalyzedType {
    pub base: AnalyzedBase,
    pub wrappers: Vec<RawWrapper>,
}

pub fn analyze_type(ty: &Type, generic_params: &[Ident]) -> Result<AnalyzedType, syn::Error> {
    let mut wrappers: Vec<RawWrapper> = Vec::new();
    let mut current_type = ty;

    // Loop to peel off wrappers in any order
    loop {
        if let Some(inner_ty) = extract_inner_type(current_type, "Option") {
            wrappers.push(RawWrapper::Option);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = extract_inner_type(current_type, "Vec") {
            wrappers.push(RawWrapper::Vec);
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

    // Bare `Duration` (no qualifier, no generic args, not a known generic
    // param) is ambiguous between `std::time::Duration` and
    // `chrono::Duration` — both crates are commonly in scope. Reject with
    // the disambiguation hint anchored at the field's type token.
    if let Type::Path(type_path) = current_type
        && type_path.qself.is_none()
        && type_path.path.segments.len() == 1
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "Duration"
        && matches!(segment.arguments, PathArguments::None)
        && !generic_params.iter().any(|p| p == &segment.ident)
    {
        return Err(syn::Error::new_spanned(
            current_type,
            "bare `Duration` is ambiguous; use `std::time::Duration` or \
             `chrono::Duration` to disambiguate",
        ));
    }

    let base = analyze_base_type(current_type, generic_params)
        .ok_or_else(|| syn::Error::new_spanned(current_type, "Unsupported field type"))?;

    Ok(AnalyzedType { base, wrappers })
}

fn analyze_base_type(ty: &Type, generic_params: &[Ident]) -> Option<AnalyzedBase> {
    if is_datetime_utc(ty) {
        return Some(AnalyzedBase::DateTimeUtc);
    }
    // Disambiguate `Duration` first (qualified-path matches) — both bases
    // share the last segment `Duration`, so naive last-segment matching is
    // insufficient. Bare `Duration` is rejected upstream in `analyze_type`,
    // not here, so the only `Duration` paths reaching this function are
    // qualified (e.g. `std::time::Duration` or `chrono::Duration`).
    if let Type::Path(type_path) = ty {
        if is_std_duration(type_path) {
            return Some(AnalyzedBase::StdDuration);
        }
        if is_chrono_duration(type_path) {
            return Some(AnalyzedBase::ChronoDuration);
        }
    }
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        let type_ident = &segment.ident;
        let has_args = !matches!(segment.arguments, PathArguments::None);
        let is_single_segment = type_path.qself.is_none() && type_path.path.segments.len() == 1;
        let base = match type_ident.to_string().as_str() {
            "String" => AnalyzedBase::String,
            "f64" => AnalyzedBase::Numeric(NumericKind::F64),
            "f32" => AnalyzedBase::Numeric(NumericKind::F32),
            "i8" => AnalyzedBase::Numeric(NumericKind::I8),
            "u8" => AnalyzedBase::Numeric(NumericKind::U8),
            "i16" => AnalyzedBase::Numeric(NumericKind::I16),
            "u16" => AnalyzedBase::Numeric(NumericKind::U16),
            "i64" => AnalyzedBase::Numeric(NumericKind::I64),
            "isize" => AnalyzedBase::Numeric(NumericKind::ISize),
            "u64" => AnalyzedBase::Numeric(NumericKind::U64),
            "usize" => AnalyzedBase::Numeric(NumericKind::USize),
            "u32" => AnalyzedBase::Numeric(NumericKind::U32),
            "i32" => AnalyzedBase::Numeric(NumericKind::I32),
            "bool" => AnalyzedBase::Bool,
            "Decimal" => AnalyzedBase::Decimal,
            // Last-segment ident matching, mirroring `is_datetime_utc`'s
            // leniency. Both `NaiveDate` and `NaiveTime` take no generic
            // arguments, so a re-export under another name (`my_crate::NaiveDate`)
            // still resolves to chrono's type at the call site; if the user's
            // type happens to share the name without sharing the API, the
            // generated `signed_duration_since` / `Timelike` calls fail at
            // compile time at the user's field site.
            "NaiveDate" if !has_args => AnalyzedBase::NaiveDate,
            "NaiveTime" if !has_args => AnalyzedBase::NaiveTime,
            _ => {
                if is_single_segment && !has_args && generic_params.iter().any(|p| p == type_ident)
                {
                    AnalyzedBase::Generic(type_ident.clone())
                } else {
                    let args = match &segment.arguments {
                        PathArguments::AngleBracketed(ab) => Some(ab.clone()),
                        _ => None,
                    };
                    AnalyzedBase::Struct(type_ident.clone(), args)
                }
            }
        };
        return Some(base);
    }
    None
}

/// Detect `std::time::Duration` or `core::time::Duration` by walking the
/// path segments. Matches when the path contains both a `time` segment and
/// a final `Duration` segment, with a leading `std` or `core` (or a
/// re-export shape that ends in `time::Duration` — same leniency as
/// `is_datetime_utc`).
fn is_std_duration(type_path: &TypePath) -> bool {
    let segs: Vec<String> = type_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    if segs.last().is_none_or(|s| s != "Duration") {
        return false;
    }
    // The reliable signal is `time` immediately preceding `Duration`. The
    // only standard-library crates that nest `Duration` under a `time`
    // module are `std::time` and `core::time`, so this is a precise match
    // for the two stdlib paths plus any re-export that preserves the tail.
    segs.iter()
        .rev()
        .nth(1)
        .is_some_and(|s| s == "time" || s == "std" || s == "core")
}

/// Detect `chrono::Duration` or `chrono::TimeDelta`. `chrono::Duration` is
/// a type alias for `chrono::TimeDelta` since chrono 0.4.30; both names
/// resolve to the same impl block, so we accept either tail. Codegen reads
/// the user's declared field-type tokens directly so type inference handles
/// the alias transparently.
fn is_chrono_duration(type_path: &TypePath) -> bool {
    let segs: Vec<String> = type_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    let last = match segs.last() {
        Some(s) if s == "Duration" || s == "TimeDelta" => s,
        _ => return false,
    };
    // For unqualified `Duration`, only accept when the path is rooted at
    // `chrono::` or contains a `chrono` segment somewhere — bare `Duration`
    // is rejected upstream, but a path like `mycrate::Duration` should not
    // route here. `TimeDelta` is chrono-specific enough that the bare-ident
    // case is unlikely to collide; still require an upstream `chrono` for
    // symmetry.
    if last == "Duration" {
        // Accept paths like `chrono::Duration` or `::chrono::Duration`.
        // Reject `std::time::Duration` (handled by `is_std_duration`),
        // `core::time::Duration`, or anything with a `time` segment in
        // the path — those are the std flavor.
        if segs.iter().any(|s| s == "time") {
            return false;
        }
        return segs.iter().any(|s| s == "chrono");
    }
    // `TimeDelta` only lives in chrono.
    segs.iter().any(|s| s == "chrono") || segs.len() == 1
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
