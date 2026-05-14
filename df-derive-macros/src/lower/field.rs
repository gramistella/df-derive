use crate::attrs::field::{FieldOverride, LeafOverride, parse_field_override};
use crate::diagnostics;
use crate::ir::{
    DateTimeUnit, DisplayBase, DurationSource, FieldIR, LeafSpec, NumericKind, StringyBase,
};
use crate::lower::tuple::{
    FieldOverrideRef, analyzed_to_tuple_element, reject_attrs_on_tuple,
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

/// Single source of truth for combining a parsed `FieldOverride` with the
/// analyzed base type into the final `LeafSpec` carried on the IR. Performs
/// base-type compatibility checks for every override variant and injects
/// the default semantics (`DateTimeToInt(Milliseconds)` for
/// `chrono::DateTime<Tz>`, `Decimal(38, 10)` for bare `Decimal` /
/// `rust_decimal::Decimal`)
/// when no override was declared.
///
/// The match is exhaustive over `(FieldOverride, AnalyzedBase)` and produces
/// one `LeafSpec` per parser-accepted pair — no `unreachable!` arms downstream.
/// `AsBinary` is handled in [`lower_field`] before this function runs
/// because it also rewrites the wrapper stack (strips the innermost `Vec`),
/// so it cannot be expressed as a `(base, override) -> LeafSpec` mapping.
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
            Some(FieldOverrideRef::Leaf {
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

/// Map an analyzed base to its default `LeafSpec` (no override declared).
/// Each base picks the parser-injected default semantics — `Milliseconds`
/// for `DateTime<Tz>`, `Nanoseconds` for `Duration`, `Decimal(38, 10)`,
/// etc. The decimal default is intentionally syntax-based and narrow: bare
/// `Decimal` and canonical `rust_decimal::Decimal` are treated as decimal
/// backends while other paths require an explicit `decimal(...)` attribute.
/// Tuple bases recurse: each element runs through the same default pipeline
/// (no field-level overrides apply at element level — the parser rejects
/// them on the parent field).
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
        // Tuple bases reach this dispatcher only when `parse_leaf_spec`'s
        // upstream `reject_attrs_on_tuple` was bypassed, which it isn't —
        // but the match must be exhaustive on `AnalyzedBase`. Surface a
        // distinct error if it ever does fire.
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
        // Tuple bases are rejected by `reject_attrs_on_tuple` before this
        // function runs. Keep this branch explicit so any bypass still emits
        // the same public diagnostic family instead of silently accepting it.
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
        // `Decimal` by name is the implicit path (`Decimal` or
        // `rust_decimal::Decimal`). Proc macros cannot resolve whether an
        // arbitrary custom type is *actually* a decimal type, so the explicit
        // `decimal(...)` attribute is the user assertion that a differently
        // named custom/generic backend should use the same `Decimal128Encode`
        // dispatch.
        AnalyzedBase::Decimal | AnalyzedBase::Struct(_) | AnalyzedBase::Generic(_) => {
            Ok(LeafSpec::Decimal { precision, scale })
        }
        _ => Err(diagnostics::decimal_wrong_base(field, field_display_name)),
    }
}

fn decimal_generic_params_for_override(
    override_: Option<&FieldOverride>,
    base: &AnalyzedBase,
) -> Vec<Ident> {
    match (override_, base) {
        (Some(FieldOverride::Leaf(LeafOverride::Decimal { .. })), AnalyzedBase::Generic(ident)) => {
            vec![ident.clone()]
        }
        _ => Vec::new(),
    }
}

fn decimal_backend_ty_for_override(
    override_: Option<&FieldOverride>,
    base: &AnalyzedBase,
) -> Option<syn::Type> {
    match (override_, base) {
        (Some(FieldOverride::Leaf(LeafOverride::Decimal { .. })), AnalyzedBase::Struct(ty)) => {
            Some(ty.clone())
        }
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

/// Validate an `as_binary` field's analyzed `(base, wrappers)` pair and,
/// on success, produce the rewritten `(LeafSpec::Binary, wrappers')` pair —
/// `wrappers'` is `wrappers` with the innermost `Vec` stripped, since the
/// `Vec<u8>` collapses into the leaf itself.
///
/// Accepts the shapes spelled out in the public docstring on `as_binary`:
/// `Vec<u8>` / `Option<Vec<u8>>` / `Vec<Vec<u8>>` / `Vec<Option<Vec<u8>>>`
/// / `Option<Vec<Vec<u8>>>` and so on. Rejects bare `u8`, `Option<u8>`,
/// `Vec<Option<u8>>` (`BinaryView` cannot carry per-byte nulls), and any
/// non-`u8` leaf with a tailored error message anchored at the field span.
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
            // Either bare `Option<u8>` (single `Option` wrapper) or any deeper
            // stack ending in `Option`-immediately-above-`u8`. Both share the
            // "no per-byte nulls" rejection.
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

/// Run the per-field pipeline (override parsing, type analysis, leaf-spec
/// resolution, wrapper normalization) and produce the corresponding `FieldIR`.
/// Named and tuple arms share this body; they only differ in how `name_ident`
/// and `field_index` are derived from the surrounding `Fields` shape.
pub fn lower_field(
    field: &syn::Field,
    name_ident: Ident,
    field_index: Option<usize>,
    struct_name: &Ident,
    generic_params: &[Ident],
) -> Result<Option<FieldIR>, syn::Error> {
    let display_name = name_ident.to_string();
    let override_ = parse_field_override(field, &display_name)?;
    let override_value = override_.as_ref().map(|override_| &override_.value);
    let override_span = override_.as_ref().map(|override_| override_.span);
    if matches!(override_value, Some(FieldOverride::Skip)) {
        return Ok(None);
    }
    let analyzed = analyze_type(&field.ty, generic_params)?;
    reject_direct_self_reference(&analyzed, &display_name, struct_name)?;
    reject_unsupported_wrapped_nested_tuples(&analyzed, &display_name)?;
    let outer_smart_ptr_depth = analyzed.outer_smart_ptr_depth;
    let decimal_generic_params =
        decimal_generic_params_for_override(override_value, &analyzed.base);
    let decimal_backend_ty = decimal_backend_ty_for_override(override_value, &analyzed.base);
    let (leaf_spec, wrapper_shape) = if matches!(override_value, Some(FieldOverride::AsBinary)) {
        // `as_binary` over a tuple is rejected here too — `parse_as_binary_shape`
        // only checks the leaf base, but the tuple itself fails the same
        // multi-column attribute rule as every other field-level attribute.
        if matches!(analyzed.base, AnalyzedBase::Tuple(_))
            && let Some(value) = override_value
            && let Some(span) = override_span
        {
            reject_attrs_on_tuple(
                field,
                &display_name,
                Some(FieldOverrideRef::Field { value, span }),
            )?;
        }
        let (leaf, trimmed) =
            parse_as_binary_shape(field, &display_name, &analyzed.base, &analyzed.wrappers)?;
        (leaf, normalize_wrappers(&trimmed))
    } else {
        let leaf = parse_leaf_spec(
            field,
            &display_name,
            override_value.and_then(FieldOverride::leaf),
            override_span,
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
