use crate::attrs::{FieldDisposition, LeafOverride, Spanned, parse_field_disposition};
use crate::ir::{
    DateTimeUnit, DisplayBase, DurationSource, FieldIR, LeafSpec, NumericKind, StringyBase,
};
use crate::lower::tuple::{
    FieldAttrRef, analyzed_to_tuple_element, reject_attrs_on_tuple,
    reject_unsupported_wrapped_nested_tuples,
};
use crate::lower::validation::reject_direct_self_reference;
use crate::lower::wrappers::normalize_wrappers;
use crate::type_analysis::{
    AnalyzedBase, DEFAULT_DATETIME_UNIT, DEFAULT_DECIMAL_PRECISION, DEFAULT_DECIMAL_SCALE,
    DEFAULT_DURATION_UNIT, RawWrapper, analyze_type,
};
use proc_macro2::Span;
use quote::ToTokens;
use syn::Ident;

use super::diagnostics;

fn parse_leaf_spec(
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
        AnalyzedBase::BorrowedBytes => Err(diagnostics::unannotated_borrowed_bytes(
            span,
            field_display_name,
            can_add_as_binary,
        )),
        AnalyzedBase::CowBytes => Err(diagnostics::unannotated_cow_bytes(
            span,
            field_display_name,
            can_add_as_binary,
        )),
        AnalyzedBase::BorrowedSlice => Err(diagnostics::borrowed_slice(span, field_display_name)),
        AnalyzedBase::CowSlice => Err(diagnostics::cow_slice(span, field_display_name)),
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
        | AnalyzedBase::Decimal => Err(diagnostics::as_str_wrong_base(field, field_display_name)),
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
        AnalyzedBase::StdDuration => Err(diagnostics::as_string_std_duration(
            field,
            field_display_name,
        )),
        AnalyzedBase::BorrowedBytes | AnalyzedBase::CowBytes => {
            Err(diagnostics::as_string_bytes(field, field_display_name))
        }
        AnalyzedBase::BorrowedSlice | AnalyzedBase::CowSlice => {
            Err(diagnostics::as_string_slice(field, field_display_name))
        }
        AnalyzedBase::Tuple(_) => Err(diagnostics::as_string_tuple(field, field_display_name)),
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
        _ => Err(diagnostics::decimal_wrong_base(field, field_display_name)),
    }
}

fn decimal_generic_params_for_override(
    override_: Option<&LeafOverride>,
    base: &AnalyzedBase,
) -> Vec<Ident> {
    match (override_, base) {
        (Some(LeafOverride::Decimal { .. }), AnalyzedBase::Generic(ident)) => vec![ident.clone()],
        _ => Vec::new(),
    }
}

fn decimal_backend_ty_for_override(
    override_: Option<&LeafOverride>,
    base: &AnalyzedBase,
) -> Option<syn::Type> {
    match (override_, base) {
        (Some(LeafOverride::Decimal { .. }), AnalyzedBase::Struct(ty)) => Some(ty.clone()),
        _ => None,
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
        AnalyzedBase::NaiveDate => {
            Err(diagnostics::time_unit_naive_date(field, field_display_name))
        }
        AnalyzedBase::NaiveTime => {
            Err(diagnostics::time_unit_naive_time(field, field_display_name))
        }
        _ => Err(diagnostics::time_unit_wrong_base(field, field_display_name)),
    }
}

fn parse_as_binary_shape(
    field: &syn::Field,
    field_display_name: &str,
    base: &AnalyzedBase,
    wrappers: &[RawWrapper],
) -> Result<(LeafSpec, Vec<RawWrapper>), syn::Error> {
    if matches!(base, AnalyzedBase::CowBytes | AnalyzedBase::BorrowedBytes) {
        return Ok((LeafSpec::Binary, wrappers.to_vec()));
    }
    if matches!(base, AnalyzedBase::BorrowedSlice) {
        return Err(diagnostics::binary_borrowed_slice(
            field,
            field_display_name,
        ));
    }
    if matches!(base, AnalyzedBase::CowSlice) {
        return Err(diagnostics::binary_cow_slice(field, field_display_name));
    }
    if !matches!(base, AnalyzedBase::Numeric(NumericKind::U8)) {
        return Err(diagnostics::binary_wrong_base(field, field_display_name));
    }
    match wrappers.last() {
        None => Err(diagnostics::bare_binary_u8(field, field_display_name)),
        Some(RawWrapper::Option) => {
            if wrappers.len() == 1 {
                Err(diagnostics::bare_binary_u8(field, field_display_name))
            } else {
                Err(diagnostics::binary_inner_option(field, field_display_name))
            }
        }
        Some(RawWrapper::Vec) => {
            let mut trimmed = wrappers.to_vec();
            trimmed.pop();
            Ok((LeafSpec::Binary, trimmed))
        }
        Some(RawWrapper::SmartPtr) => {
            Err(diagnostics::binary_wrong_base(field, field_display_name))
        }
    }
}

pub fn lower_field(
    field: &syn::Field,
    name_ident: Ident,
    field_index: Option<usize>,
    struct_name: &Ident,
    generic_params: &[Ident],
) -> Result<Option<FieldIR>, syn::Error> {
    let display_name = name_ident.to_string();
    let disposition = parse_field_disposition(field, &display_name)?;
    if matches!(disposition, FieldDisposition::Skip) {
        return Ok(None);
    }
    let analyzed = analyze_type(&field.ty, generic_params)?;
    reject_direct_self_reference(&analyzed, &display_name, struct_name)?;
    reject_unsupported_wrapped_nested_tuples(&analyzed, &display_name)?;
    let outer_smart_ptr_depth = analyzed.outer_smart_ptr_depth;
    let leaf_override: Option<&Spanned<LeafOverride>> = match &disposition {
        FieldDisposition::Include { leaf_override } => leaf_override.as_ref(),
        FieldDisposition::Skip => unreachable!("skip disposition returned before type analysis"),
        FieldDisposition::Binary { .. } => None,
    };
    let leaf_override_value = leaf_override.map(|override_| &override_.value);
    let decimal_generic_params =
        decimal_generic_params_for_override(leaf_override_value, &analyzed.base);
    let decimal_backend_ty = decimal_backend_ty_for_override(leaf_override_value, &analyzed.base);
    let (leaf_spec, wrapper_shape) = if let FieldDisposition::Binary { span } = &disposition {
        if matches!(analyzed.base, AnalyzedBase::Tuple(_)) {
            reject_attrs_on_tuple(
                field,
                &display_name,
                Some(FieldAttrRef::Binary { span: *span }),
            )?;
        }
        let (leaf, trimmed) =
            parse_as_binary_shape(field, &display_name, &analyzed.base, &analyzed.wrappers)?;
        (leaf, normalize_wrappers(&trimmed))
    } else {
        let leaf_override_span = leaf_override.map(|override_| override_.span);
        let leaf = parse_leaf_spec(
            field,
            &display_name,
            leaf_override_value,
            leaf_override_span,
            analyzed.base,
        )?;
        (leaf, normalize_wrappers(&analyzed.wrappers))
    };
    Ok(Some(FieldIR {
        name: name_ident,
        field_index,
        leaf_spec,
        wrapper_shape,
        decimal_generic_params,
        decimal_backend_ty,
        outer_smart_ptr_depth,
    }))
}
