use crate::attrs::LeafOverride;
use crate::ir::{DateTimeUnit, DisplayBase, DurationSource, LeafSpec, StringyBase};
use crate::lower::tuple::{FieldAttrRef, analyzed_to_tuple_element, reject_attrs_on_tuple};
use crate::type_analysis::{
    AnalyzedBase, DEFAULT_DATETIME_UNIT, DEFAULT_DECIMAL_PRECISION, DEFAULT_DECIMAL_SCALE,
    DEFAULT_DURATION_UNIT,
};
use proc_macro2::Span;
use quote::ToTokens;

use super::errors;

pub(super) fn parse_leaf_spec(
    field: &syn::Field,
    field_display_name: &str,
    override_: Option<&LeafOverride>,
    override_span: Option<Span>,
    base: AnalyzedBase,
) -> Result<LeafSpec, syn::Error> {
    if let AnalyzedBase::Tuple(_) = &base
        && let Some(override_) = override_
        && let Some(span) = override_span
    {
        reject_attrs_on_tuple(
            field,
            field_display_name,
            Some(FieldAttrRef::Leaf {
                value: override_,
                span,
            }),
        )?;
    }
    match override_ {
        None => default_leaf_for_base(field, field_display_name, base, true),
        Some(LeafOverride::AsString) => parse_leaf_as_string(field, field_display_name, &base),
        Some(LeafOverride::AsStr) => parse_leaf_as_str(field, field_display_name, base),
        Some(LeafOverride::Decimal { precision, scale }) => {
            parse_leaf_decimal(field, field_display_name, &base, *precision, *scale)
        }
        Some(LeafOverride::TimeUnit(unit)) => {
            parse_leaf_time_unit(field, field_display_name, &base, *unit)
        }
    }
}

pub(super) fn default_leaf_for_base<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
    base: AnalyzedBase,
    can_add_as_binary: bool,
) -> Result<LeafSpec, syn::Error> {
    match base {
        AnalyzedBase::Numeric(kind) => Ok(LeafSpec::Numeric(kind)),
        AnalyzedBase::String => Ok(LeafSpec::String),
        AnalyzedBase::BorrowedStr => Ok(LeafSpec::AsStr(StringyBase::BorrowedStr)),
        AnalyzedBase::CowStr => Ok(LeafSpec::AsStr(StringyBase::CowStr)),
        AnalyzedBase::BorrowedBytes => Err(errors::unannotated_borrowed_bytes(
            span,
            field_display_name,
            can_add_as_binary,
        )),
        AnalyzedBase::CowBytes => Err(errors::unannotated_cow_bytes(
            span,
            field_display_name,
            can_add_as_binary,
        )),
        AnalyzedBase::BorrowedSlice => Err(errors::borrowed_slice(span, field_display_name)),
        AnalyzedBase::CowSlice => Err(errors::cow_slice(span, field_display_name)),
        AnalyzedBase::Bool => Ok(LeafSpec::Bool),
        AnalyzedBase::DateTimeTz => Ok(LeafSpec::DateTime(DEFAULT_DATETIME_UNIT)),
        AnalyzedBase::NaiveDate => Ok(LeafSpec::NaiveDate),
        AnalyzedBase::NaiveTime => Ok(LeafSpec::NaiveTime),
        AnalyzedBase::NaiveDateTime => Ok(LeafSpec::NaiveDateTime(DEFAULT_DATETIME_UNIT)),
        AnalyzedBase::StdDuration => Ok(LeafSpec::Duration {
            unit: DEFAULT_DURATION_UNIT,
            source: DurationSource::Std,
        }),
        AnalyzedBase::ChronoDuration => Ok(LeafSpec::Duration {
            unit: DEFAULT_DURATION_UNIT,
            source: DurationSource::Chrono,
        }),
        AnalyzedBase::Decimal => Ok(LeafSpec::Decimal {
            precision: DEFAULT_DECIMAL_PRECISION,
            scale: DEFAULT_DECIMAL_SCALE,
        }),
        AnalyzedBase::Struct(ty) => Ok(LeafSpec::Struct(ty)),
        AnalyzedBase::Generic(ident) => Ok(LeafSpec::Generic(ident)),
        AnalyzedBase::Tuple(elements) => {
            let lowered: Result<Vec<_>, _> = elements
                .into_iter()
                .map(|element| analyzed_to_tuple_element(element, field_display_name))
                .collect();
            Ok(LeafSpec::Tuple(lowered?))
        }
    }
}

fn parse_leaf_as_str(
    field: &syn::Field,
    field_display_name: &str,
    base: AnalyzedBase,
) -> Result<LeafSpec, syn::Error> {
    match base {
        AnalyzedBase::String => Ok(LeafSpec::AsStr(StringyBase::String)),
        AnalyzedBase::BorrowedStr => Ok(LeafSpec::AsStr(StringyBase::BorrowedStr)),
        AnalyzedBase::CowStr => Ok(LeafSpec::AsStr(StringyBase::CowStr)),
        AnalyzedBase::Struct(ty) => Ok(LeafSpec::AsStr(StringyBase::Struct(ty))),
        AnalyzedBase::Generic(ident) => Ok(LeafSpec::AsStr(StringyBase::Generic(ident))),
        AnalyzedBase::Tuple(_)
        | AnalyzedBase::Numeric(_)
        | AnalyzedBase::BorrowedBytes
        | AnalyzedBase::CowBytes
        | AnalyzedBase::BorrowedSlice
        | AnalyzedBase::CowSlice
        | AnalyzedBase::Bool
        | AnalyzedBase::DateTimeTz
        | AnalyzedBase::NaiveDate
        | AnalyzedBase::NaiveTime
        | AnalyzedBase::NaiveDateTime
        | AnalyzedBase::StdDuration
        | AnalyzedBase::ChronoDuration
        | AnalyzedBase::Decimal => Err(errors::as_str_wrong_base(field, field_display_name)),
    }
}

fn parse_leaf_as_string(
    field: &syn::Field,
    field_display_name: &str,
    base: &AnalyzedBase,
) -> Result<LeafSpec, syn::Error> {
    display_base_for_as_string(field, field_display_name, base).map(LeafSpec::AsString)
}

fn display_base_for_as_string(
    field: &syn::Field,
    field_display_name: &str,
    base: &AnalyzedBase,
) -> Result<DisplayBase, syn::Error> {
    match base {
        AnalyzedBase::Numeric(_)
        | AnalyzedBase::String
        | AnalyzedBase::BorrowedStr
        | AnalyzedBase::CowStr
        | AnalyzedBase::Bool
        | AnalyzedBase::DateTimeTz
        | AnalyzedBase::NaiveDate
        | AnalyzedBase::NaiveTime
        | AnalyzedBase::NaiveDateTime
        | AnalyzedBase::ChronoDuration
        | AnalyzedBase::Decimal => Ok(DisplayBase::Inherent),
        AnalyzedBase::Struct(ty) => Ok(DisplayBase::Struct(ty.clone())),
        AnalyzedBase::Generic(ident) => Ok(DisplayBase::Generic(ident.clone())),
        AnalyzedBase::StdDuration => Err(errors::as_string_std_duration(field, field_display_name)),
        AnalyzedBase::BorrowedBytes | AnalyzedBase::CowBytes => {
            Err(errors::as_string_bytes(field, field_display_name))
        }
        AnalyzedBase::BorrowedSlice | AnalyzedBase::CowSlice => {
            Err(errors::as_string_slice(field, field_display_name))
        }
        AnalyzedBase::Tuple(_) => Err(errors::as_string_tuple(field, field_display_name)),
    }
}

fn parse_leaf_decimal(
    field: &syn::Field,
    field_display_name: &str,
    base: &AnalyzedBase,
    precision: u8,
    scale: u8,
) -> Result<LeafSpec, syn::Error> {
    match base {
        AnalyzedBase::Decimal | AnalyzedBase::Struct(_) | AnalyzedBase::Generic(_) => {
            Ok(LeafSpec::Decimal { precision, scale })
        }
        _ => Err(errors::decimal_wrong_base(field, field_display_name)),
    }
}

fn parse_leaf_time_unit(
    field: &syn::Field,
    field_display_name: &str,
    base: &AnalyzedBase,
    unit: DateTimeUnit,
) -> Result<LeafSpec, syn::Error> {
    match base {
        AnalyzedBase::DateTimeTz => Ok(LeafSpec::DateTime(unit)),
        AnalyzedBase::NaiveDateTime => Ok(LeafSpec::NaiveDateTime(unit)),
        AnalyzedBase::StdDuration => Ok(LeafSpec::Duration {
            unit,
            source: DurationSource::Std,
        }),
        AnalyzedBase::ChronoDuration => Ok(LeafSpec::Duration {
            unit,
            source: DurationSource::Chrono,
        }),
        AnalyzedBase::NaiveDate => Err(errors::time_unit_naive_date(field, field_display_name)),
        AnalyzedBase::NaiveTime => Err(errors::time_unit_naive_time(field, field_display_name)),
        _ => Err(errors::time_unit_wrong_base(field, field_display_name)),
    }
}
