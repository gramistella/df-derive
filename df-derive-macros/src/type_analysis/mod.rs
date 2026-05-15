mod diagnostics;
mod known_types;
mod path_match;
mod wrappers;

use crate::ir::{DateTimeUnit, NumericKind};
use syn::{Ident, PathArguments, Type};

use diagnostics::{
    reject_bare_duration, reject_bare_unsized_leaf, reject_unsupported_collection_type,
};
use known_types::classify_known_base;
use wrappers::{analyze_cow_base, borrowed_reference_base, peel_type_wrappers};

/// Default `Datetime` precision for `chrono::DateTime<Tz>` and
/// `chrono::NaiveDateTime` fields without an explicit `time_unit` override.
pub const DEFAULT_DATETIME_UNIT: DateTimeUnit = DateTimeUnit::Milliseconds;
/// Default `Duration` precision for `std::time::Duration` and
/// `chrono::Duration` (`chrono::TimeDelta`) fields without an explicit
/// `time_unit` override.
pub const DEFAULT_DURATION_UNIT: DateTimeUnit = DateTimeUnit::Nanoseconds;
/// Default `Decimal(precision, scale)` for bare `Decimal` or
/// `rust_decimal::Decimal` fields without an explicit `decimal(...)` override.
pub const DEFAULT_DECIMAL_PRECISION: u8 = 38;
/// Default scale paired with `DEFAULT_DECIMAL_PRECISION`.
pub const DEFAULT_DECIMAL_SCALE: u8 = 10;

/// Raw wrapper position before normalization. The parser collapses these
/// into a `WrapperShape` after type analysis.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RawWrapper {
    Option,
    Vec,
    SmartPtr,
}

/// Analyzed base type before parser-time override fusion into a `LeafSpec`.
#[derive(Clone)]
pub enum AnalyzedBase {
    Numeric(NumericKind),
    String,
    /// `&str` — kept as a semantic string leaf instead of peeling to
    /// unsized `str`, mirroring `Cow<'_, str>`.
    BorrowedStr,
    /// `Cow<'_, str>` — kept as a semantic string leaf instead of peeling to
    /// unsized `str`.
    CowStr,
    /// `&[u8]` — supported only with `#[df_derive(as_binary)]`.
    BorrowedBytes,
    /// `Cow<'_, [u8]>` — supported only with `#[df_derive(as_binary)]`.
    CowBytes,
    /// `&[T]` for non-`u8` element types.
    BorrowedSlice,
    /// `Cow<'_, [T]>` for non-`u8` element types.
    CowSlice,
    Bool,
    /// `chrono::DateTime<Tz>`.
    DateTimeTz,
    /// `chrono::NaiveDate`.
    NaiveDate,
    /// `chrono::NaiveTime`.
    NaiveTime,
    /// `chrono::NaiveDateTime`.
    NaiveDateTime,
    /// Exactly `std::time::Duration` or `core::time::Duration`.
    StdDuration,
    /// `chrono::Duration` or `chrono::TimeDelta`.
    ChronoDuration,
    /// Bare `Decimal` or canonical `rust_decimal::Decimal`.
    Decimal,
    /// Concrete user-defined struct type as written at the field use site.
    Struct(Type),
    /// Generic type parameter declared on the enclosing struct.
    Generic(Ident),
    /// Tuple-typed base, with each element recursively analyzed.
    Tuple(Vec<AnalyzedType>),
}

#[derive(Clone)]
pub struct AnalyzedType {
    pub base: AnalyzedBase,
    pub wrappers: Vec<RawWrapper>,
    /// Original syntactic type token for this analyzed type.
    pub field_ty: syn::Type,
    /// Number of transparent pointer layers (`Box` / `Rc` / `Arc` / `Cow` /
    /// borrowed references) peeled off above the first semantic wrapper.
    pub outer_smart_ptr_depth: usize,
}

fn bare_generic_param_ident(ty: &Type, generic_params: &[Ident]) -> Option<Ident> {
    let Type::Path(type_path) = ty else {
        return None;
    };

    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }

    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }

    generic_params
        .iter()
        .any(|param| param == &segment.ident)
        .then(|| segment.ident.clone())
}

pub fn analyze_type(ty: &Type, generic_params: &[Ident]) -> Result<AnalyzedType, syn::Error> {
    let peeled = peel_type_wrappers(ty)?;

    if bare_generic_param_ident(peeled.current_type, generic_params).is_none() {
        reject_unsupported_collection_type(peeled.current_type)?;
        reject_bare_duration(peeled.current_type, generic_params)?;
        reject_bare_unsized_leaf(peeled.current_type)?;
    }

    let base = analyze_base_type(peeled.current_type, generic_params)?;

    Ok(AnalyzedType {
        base,
        wrappers: peeled.wrappers,
        outer_smart_ptr_depth: peeled.outer_smart_ptr_depth,
        field_ty: ty.clone(),
    })
}

fn analyze_base_type(ty: &Type, generic_params: &[Ident]) -> Result<AnalyzedBase, syn::Error> {
    if let Some(tuple) = analyze_tuple_base(ty, generic_params)? {
        return Ok(tuple);
    }

    if let Type::Reference(reference) = ty
        && let Some(base) = borrowed_reference_base(reference)
    {
        return Ok(base);
    }

    if let Some(ident) = bare_generic_param_ident(ty, generic_params) {
        return Ok(AnalyzedBase::Generic(ident));
    }

    if let Type::Path(type_path) = ty {
        if let Some(base) = analyze_cow_base(type_path) {
            return Ok(base);
        }
        if let Some(known) = classify_known_base(type_path) {
            return Ok(known.into_analyzed_base());
        }
        return Ok(AnalyzedBase::Struct(ty.clone()));
    }

    Err(syn::Error::new_spanned(ty, "Unsupported field type"))
}

fn analyze_tuple_base(
    ty: &Type,
    generic_params: &[Ident],
) -> Result<Option<AnalyzedBase>, syn::Error> {
    let Type::Tuple(tuple) = ty else {
        return Ok(None);
    };

    if tuple.elems.is_empty() {
        return Err(syn::Error::new_spanned(
            ty,
            "df-derive does not support direct unit-typed (`()`) fields; \
             they would contribute zero columns. Remove the field, replace \
             it with a non-unit type, or use a generic payload such as \
             `field: M` with `M = ()`.",
        ));
    }

    let mut elements: Vec<AnalyzedType> = Vec::with_capacity(tuple.elems.len());
    for elem in &tuple.elems {
        elements.push(analyze_type(elem, generic_params)?);
    }
    Ok(Some(AnalyzedBase::Tuple(elements)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;

    fn analyze(ty: &Type) -> AnalyzedType {
        analyze_type(ty, &[]).expect("type should analyze")
    }

    fn analyze_with_generics(ty: &Type, generic_params: &[&str]) -> AnalyzedType {
        let generic_params: Vec<Ident> = generic_params
            .iter()
            .map(|param| Ident::new(param, Span::call_site()))
            .collect();
        analyze_type(ty, &generic_params).expect("type should analyze")
    }

    #[test]
    fn recognizes_string_paths() {
        for ty in [
            syn::parse_quote!(String),
            syn::parse_quote!(std::string::String),
            syn::parse_quote!(alloc::string::String),
        ] {
            assert!(matches!(analyze(&ty).base, AnalyzedBase::String));
        }
    }

    #[test]
    fn recognizes_borrowed_and_cow_unsized_bases() {
        assert!(matches!(
            analyze(&syn::parse_quote!(&'a str)).base,
            AnalyzedBase::BorrowedStr
        ));
        assert!(matches!(
            analyze(&syn::parse_quote!(std::borrow::Cow<'a, str>)).base,
            AnalyzedBase::CowStr
        ));
        assert!(matches!(
            analyze(&syn::parse_quote!(&'a [u8])).base,
            AnalyzedBase::BorrowedBytes
        ));
        assert!(matches!(
            analyze(&syn::parse_quote!(std::borrow::Cow<'a, [u8]>)).base,
            AnalyzedBase::CowBytes
        ));
        assert!(matches!(
            analyze_with_generics(&syn::parse_quote!(&'a [T]), &["T"]).base,
            AnalyzedBase::BorrowedSlice
        ));
    }

    #[test]
    fn recognizes_duration_families() {
        assert!(matches!(
            analyze(&syn::parse_quote!(std::time::Duration)).base,
            AnalyzedBase::StdDuration
        ));
        assert!(matches!(
            analyze(&syn::parse_quote!(core::time::Duration)).base,
            AnalyzedBase::StdDuration
        ));
        assert!(matches!(
            analyze(&syn::parse_quote!(chrono::Duration)).base,
            AnalyzedBase::ChronoDuration
        ));
        assert!(matches!(
            analyze(&syn::parse_quote!(chrono::TimeDelta)).base,
            AnalyzedBase::ChronoDuration
        ));
    }

    #[test]
    fn records_wrapper_stack_order_from_outer_to_inner() {
        let opt_vec_u8 = analyze(&syn::parse_quote!(Option<Vec<u8>>));
        assert!(matches!(
            opt_vec_u8.wrappers.as_slice(),
            [RawWrapper::Option, RawWrapper::Vec]
        ));
        assert!(matches!(
            opt_vec_u8.base,
            AnalyzedBase::Numeric(NumericKind::U8)
        ));

        let nested = analyze_with_generics(&syn::parse_quote!(Vec<Option<Vec<T>>>), &["T"]);
        assert!(matches!(
            nested.wrappers.as_slice(),
            [RawWrapper::Vec, RawWrapper::Option, RawWrapper::Vec]
        ));
        assert!(matches!(nested.base, AnalyzedBase::Generic(ref ident) if ident == "T"));
    }

    #[test]
    fn does_not_peel_wrappers_with_extra_generic_arguments() {
        let analyzed = analyze_with_generics(&syn::parse_quote!(Option<T, U>), &["T", "U"]);

        assert!(analyzed.wrappers.is_empty());
        assert!(matches!(analyzed.base, AnalyzedBase::Struct(_)));
    }

    #[test]
    fn recursively_analyzes_tuple_elements() {
        let analyzed = analyze(&syn::parse_quote!((i32, String)));
        let AnalyzedBase::Tuple(elements) = analyzed.base else {
            panic!("tuple should analyze as tuple base");
        };

        assert_eq!(elements.len(), 2);
        assert!(matches!(
            elements[0].base,
            AnalyzedBase::Numeric(NumericKind::I32)
        ));
        assert!(matches!(elements[1].base, AnalyzedBase::String));
    }
}
