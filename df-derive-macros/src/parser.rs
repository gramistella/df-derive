use crate::ir::{
    AccessChain, AccessStep, DateTimeUnit, DisplayBase, DurationSource, FieldIR, LeafShape,
    LeafSpec, NonEmpty, NumericKind, StringyBase, StructIR, TupleElement, VecLayerSpec, VecLayers,
    WrapperShape,
};
use crate::type_analysis::{
    AnalyzedBase, AnalyzedType, DEFAULT_DATETIME_UNIT, DEFAULT_DECIMAL_PRECISION,
    DEFAULT_DECIMAL_SCALE, DEFAULT_DURATION_UNIT, RawWrapper, analyze_type,
};
use proc_macro2::Span;
use quote::{ToTokens, format_ident};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Fields, Ident};

/// Leaf-level conversion override declared via `#[df_derive(...)]`.
/// `skip` and `as_binary` are field dispositions, so they are deliberately
/// not representable here and cannot reach leaf parsing.
enum LeafOverride {
    AsStr,
    AsString,
    Decimal { precision: u8, scale: u8 },
    TimeUnit(DateTimeUnit),
}

/// Mutually-exclusive field-level override declared via `#[df_derive(...)]`.
enum FieldOverride {
    Skip,
    AsBinary,
    Leaf(LeafOverride),
}

impl FieldOverride {
    const fn leaf(&self) -> Option<&LeafOverride> {
        match self {
            Self::Leaf(override_) => Some(override_),
            Self::Skip | Self::AsBinary => None,
        }
    }
}

fn duplicate_decimal_key_error(
    key: &'static str,
    existing: (u8, Span),
    incoming_value: u8,
    incoming_span: Span,
) -> syn::Error {
    let (existing_value, existing_span) = existing;
    let message = if existing_value == incoming_value {
        format!("`decimal(...)` declares duplicate `{key}` key; remove one")
    } else {
        format!(
            "`decimal(...)` declares duplicate `{key}` keys with different values; \
             first is `{existing_value}`, second is `{incoming_value}`; pick one"
        )
    };

    let mut error = syn::Error::new(incoming_span, message);
    error.combine(syn::Error::new(
        existing_span,
        format!("first `{key}` key declared here"),
    ));
    error
}

fn parse_decimal_attr(meta: &syn::meta::ParseNestedMeta<'_>) -> Result<(u8, u8), syn::Error> {
    let mut precision: Option<(u8, Span)> = None;
    let mut scale: Option<(u8, Span)> = None;
    meta.parse_nested_meta(|sub| {
        if sub.path.is_ident("precision") {
            let key_span = sub.path.span();
            let lit: syn::LitInt = sub.value()?.parse()?;
            let value: u8 = lit.base10_parse().map_err(|_| {
                sub.error("decimal precision must fit in u8 (Polars requires 1..=38)")
            })?;
            if !(1..=38).contains(&value) {
                return Err(sub.error(format!(
                    "decimal precision must satisfy 1 <= precision <= 38 (got {value})"
                )));
            }
            if let Some(existing) = precision {
                return Err(duplicate_decimal_key_error(
                    "precision",
                    existing,
                    value,
                    key_span,
                ));
            }
            precision = Some((value, key_span));
            Ok(())
        } else if sub.path.is_ident("scale") {
            let key_span = sub.path.span();
            let lit: syn::LitInt = sub.value()?.parse()?;
            let value: u8 = lit
                .base10_parse()
                .map_err(|_| sub.error("decimal scale must fit in u8 (max 38)"))?;
            if value > 38 {
                return Err(sub.error(format!("decimal scale must be <= 38 (got {value})")));
            }
            if let Some(existing) = scale {
                return Err(duplicate_decimal_key_error(
                    "scale", existing, value, key_span,
                ));
            }
            scale = Some((value, key_span));
            Ok(())
        } else {
            Err(sub.error(
                "unknown key inside `decimal(...)`; expected `precision = N` or `scale = N`",
            ))
        }
    })?;
    let p = precision
        .map(|(value, _)| value)
        .ok_or_else(|| meta.error("`decimal(...)` requires `precision = N`"))?;
    let s = scale
        .map(|(value, _)| value)
        .ok_or_else(|| meta.error("`decimal(...)` requires `scale = N`"))?;
    if s > p {
        return Err(meta.error(format!("decimal scale ({s}) cannot exceed precision ({p})")));
    }
    Ok((p, s))
}

fn parse_time_unit_attr(meta: &syn::meta::ParseNestedMeta<'_>) -> Result<DateTimeUnit, syn::Error> {
    let lit: syn::LitStr = meta.value()?.parse()?;
    match lit.value().as_str() {
        "ms" => Ok(DateTimeUnit::Milliseconds),
        "us" => Ok(DateTimeUnit::Microseconds),
        "ns" => Ok(DateTimeUnit::Nanoseconds),
        other => Err(syn::Error::new_spanned(
            &lit,
            format!("invalid `time_unit` value `{other}`; expected one of \"ms\", \"us\", \"ns\""),
        )),
    }
}

/// Build the conflict error when a second mutually-exclusive override key is
/// encountered after the first has already been set. Spans on the field so
/// the message lands on the entire `#[df_derive(...)]` plus declaration block.
fn override_conflict(
    field: &syn::Field,
    field_display_name: &str,
    existing: &FieldOverride,
    incoming: &FieldOverride,
) -> syn::Error {
    let message = match (existing, incoming) {
        (FieldOverride::Leaf(LeafOverride::AsStr), FieldOverride::Leaf(LeafOverride::AsString))
        | (FieldOverride::Leaf(LeafOverride::AsString), FieldOverride::Leaf(LeafOverride::AsStr)) =>
        {
            format!(
                "field `{field_display_name}` has both `as_str` and `as_string`; \
             pick one — `as_str` borrows via `AsRef<str>` without formatting, \
             `as_string` formats via `Display` into a reused scratch buffer"
            )
        }
        (
            FieldOverride::Leaf(LeafOverride::Decimal { .. }),
            FieldOverride::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
        )
        | (
            FieldOverride::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
            FieldOverride::Leaf(LeafOverride::Decimal { .. }),
        ) => {
            format!(
                "field `{field_display_name}` combines `decimal(...)` with `as_str`/`as_string`; \
                 `as_str`/`as_string` produce a String column, so the `decimal(...)` \
                 dtype override has no effect — drop one"
            )
        }
        (
            FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
            FieldOverride::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
        )
        | (
            FieldOverride::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
            FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
        ) => {
            format!(
                "field `{field_display_name}` combines `time_unit = \"...\"` with \
                 `as_str`/`as_string`; the latter produces a String column, so the \
                 `time_unit` override has no effect — drop one"
            )
        }
        (
            FieldOverride::Leaf(LeafOverride::Decimal { .. }),
            FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
        )
        | (
            FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
            FieldOverride::Leaf(LeafOverride::Decimal { .. }),
        ) => format!(
            "field `{field_display_name}` combines `decimal(...)` with `time_unit = \"...\"`; \
             pick one — `decimal(...)` applies to decimal backend candidates, \
             `time_unit` only applies to `chrono::DateTime<Tz>`, \
             `chrono::NaiveDateTime`, `std::time::Duration`, \
             `core::time::Duration`, or `chrono::Duration`"
        ),
        (FieldOverride::Skip, _) | (_, FieldOverride::Skip) => format!(
            "field `{field_display_name}` combines `skip` with another field attribute; \
             `skip` omits the field entirely, so conversion attributes have no effect; drop one"
        ),
        (FieldOverride::AsBinary, _) | (_, FieldOverride::AsBinary) => format!(
            "field `{field_display_name}` combines `as_binary` with another override; \
             `as_binary` produces a Binary column over a `Vec<u8>` shape and is \
             mutually exclusive with `as_str`, `as_string`, `decimal(...)`, and \
             `time_unit = \"...\"` — drop one"
        ),
        (FieldOverride::Leaf(LeafOverride::AsStr), FieldOverride::Leaf(LeafOverride::AsStr))
        | (
            FieldOverride::Leaf(LeafOverride::AsString),
            FieldOverride::Leaf(LeafOverride::AsString),
        )
        | (
            FieldOverride::Leaf(LeafOverride::Decimal { .. }),
            FieldOverride::Leaf(LeafOverride::Decimal { .. }),
        )
        | (
            FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
            FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
        ) => format!(
            "field `{field_display_name}` declares duplicate `{}` override; remove one",
            override_key(incoming)
        ),
    };
    syn::Error::new_spanned(field, message)
}

const fn override_key(override_: &FieldOverride) -> &'static str {
    match override_ {
        FieldOverride::Skip => "skip",
        FieldOverride::AsBinary => "as_binary",
        FieldOverride::Leaf(LeafOverride::AsStr) => "as_str",
        FieldOverride::Leaf(LeafOverride::AsString) => "as_string",
        FieldOverride::Leaf(LeafOverride::Decimal { .. }) => "decimal(...)",
        FieldOverride::Leaf(LeafOverride::TimeUnit(_)) => "time_unit",
    }
}

const fn time_unit_attr_value(unit: DateTimeUnit) -> &'static str {
    match unit {
        DateTimeUnit::Milliseconds => "ms",
        DateTimeUnit::Microseconds => "us",
        DateTimeUnit::Nanoseconds => "ns",
    }
}

fn duplicate_override_conflict(
    field_display_name: &str,
    existing: &FieldOverride,
    incoming: &FieldOverride,
    existing_span: Span,
    incoming_span: Span,
) -> syn::Error {
    let key = override_key(incoming);
    let message = match (existing, incoming) {
        (
            FieldOverride::Leaf(LeafOverride::Decimal {
                precision: existing_precision,
                scale: existing_scale,
            }),
            FieldOverride::Leaf(LeafOverride::Decimal {
                precision: incoming_precision,
                scale: incoming_scale,
            }),
        ) if existing_precision != incoming_precision || existing_scale != incoming_scale => {
            format!(
                "field `{field_display_name}` declares duplicate `decimal(...)` overrides with \
                 different values; first is `precision = {existing_precision}, scale = {existing_scale}`, \
                 second is `precision = {incoming_precision}, scale = {incoming_scale}`; pick one"
            )
        }
        (
            FieldOverride::Leaf(LeafOverride::TimeUnit(existing_unit)),
            FieldOverride::Leaf(LeafOverride::TimeUnit(incoming_unit)),
        ) if existing_unit != incoming_unit => {
            let existing_unit = time_unit_attr_value(*existing_unit);
            let incoming_unit = time_unit_attr_value(*incoming_unit);
            format!(
                "field `{field_display_name}` declares duplicate `time_unit` overrides with \
                 different values; first is `{existing_unit}`, second is `{incoming_unit}`; pick one"
            )
        }
        _ => {
            format!("field `{field_display_name}` declares duplicate `{key}` override; remove one")
        }
    };

    let mut error = syn::Error::new(incoming_span, message);
    error.combine(syn::Error::new(
        existing_span,
        format!("first `{key}` override declared here"),
    ));
    error
}

/// Set `override_` to `incoming` only if no override has been declared yet;
/// otherwise emit the conflict error for the (existing, incoming) pair. Same-key
/// repeats (`#[df_derive(as_str, as_str)]`, two `decimal(...)` blocks) are
/// rejected so users do not accidentally rely on declaration order.
fn set_override(
    field: &syn::Field,
    field_display_name: &str,
    override_: &mut Option<(FieldOverride, Span)>,
    incoming: FieldOverride,
    incoming_span: Span,
) -> Result<(), syn::Error> {
    match override_ {
        None => {
            *override_ = Some((incoming, incoming_span));
            Ok(())
        }
        Some((existing, existing_span)) if same_override_key(existing, &incoming) => {
            Err(duplicate_override_conflict(
                field_display_name,
                existing,
                &incoming,
                *existing_span,
                incoming_span,
            ))
        }
        Some((existing, _)) => Err(override_conflict(
            field,
            field_display_name,
            existing,
            &incoming,
        )),
    }
}

const fn same_override_key(existing: &FieldOverride, incoming: &FieldOverride) -> bool {
    matches!(
        (existing, incoming),
        (FieldOverride::Skip, FieldOverride::Skip)
            | (FieldOverride::AsBinary, FieldOverride::AsBinary)
            | (
                FieldOverride::Leaf(LeafOverride::AsStr),
                FieldOverride::Leaf(LeafOverride::AsStr),
            )
            | (
                FieldOverride::Leaf(LeafOverride::AsString),
                FieldOverride::Leaf(LeafOverride::AsString),
            )
            | (
                FieldOverride::Leaf(LeafOverride::Decimal { .. }),
                FieldOverride::Leaf(LeafOverride::Decimal { .. }),
            )
            | (
                FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
                FieldOverride::Leaf(LeafOverride::TimeUnit(_)),
            )
    )
}

fn parse_field_override(
    field: &syn::Field,
    field_display_name: &str,
) -> Result<Option<FieldOverride>, syn::Error> {
    let mut override_: Option<(FieldOverride, Span)> = None;
    for attr in &field.attrs {
        if attr.path().is_ident("df_derive") {
            attr.parse_nested_meta(|meta| {
                let incoming_span = meta.path.span();
                if meta.path.is_ident("skip") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::Skip,
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_string") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::Leaf(LeafOverride::AsString),
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_str") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::Leaf(LeafOverride::AsStr),
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_binary") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::AsBinary,
                        incoming_span,
                    )
                } else if meta.path.is_ident("decimal") {
                    let (precision, scale) = parse_decimal_attr(&meta)?;
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::Leaf(LeafOverride::Decimal { precision, scale }),
                        incoming_span,
                    )
                } else if meta.path.is_ident("time_unit") {
                    let unit = parse_time_unit_attr(&meta)?;
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::Leaf(LeafOverride::TimeUnit(unit)),
                        incoming_span,
                    )
                } else {
                    Err(meta.error(
                        "unknown key in #[df_derive(...)] field attribute; expected `skip`, `as_str`, `as_string`, `as_binary`, `decimal(precision = N, scale = N)`, or `time_unit = \"ms\"|\"us\"|\"ns\"`",
                    ))
                }
            })?;
        }
    }
    Ok(override_.map(|(override_, _)| override_))
}

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
/// `AsBinary` is handled in [`process_field`] before this function runs
/// because it also rewrites the wrapper stack (strips the innermost `Vec`),
/// so it cannot be expressed as a `(base, override) -> LeafSpec` mapping.
fn parse_leaf_spec(
    field: &syn::Field,
    field_display_name: &str,
    override_: Option<&LeafOverride>,
    base: AnalyzedBase,
) -> Result<LeafSpec, syn::Error> {
    if let AnalyzedBase::Tuple(_) = &base {
        reject_attrs_on_tuple(
            field,
            field_display_name,
            override_.map(FieldOverrideRef::Leaf),
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

#[derive(Clone, Copy)]
enum FieldOverrideRef<'a> {
    Field(&'a FieldOverride),
    Leaf(&'a LeafOverride),
}

/// Reject every field-level override on a tuple-typed field with a
/// per-attribute message. Attributes apply to a single column's leaf
/// classification and have no per-element selector, so `as_str` / `as_string`
/// / `as_binary` / `decimal(...)` / `time_unit = "..."` over a tuple is
/// always ambiguous. The fix is to hoist the tuple into a named struct
/// where per-element attributes can be applied at field level.
fn reject_attrs_on_tuple(
    field: &syn::Field,
    field_display_name: &str,
    override_: Option<FieldOverrideRef<'_>>,
) -> Result<(), syn::Error> {
    let attr = match override_ {
        None | Some(FieldOverrideRef::Field(FieldOverride::Skip)) => return Ok(()),
        Some(FieldOverrideRef::Field(FieldOverride::AsBinary)) => "as_binary",
        Some(
            FieldOverrideRef::Field(FieldOverride::Leaf(override_))
            | FieldOverrideRef::Leaf(override_),
        ) => match override_ {
            LeafOverride::AsStr => "as_str",
            LeafOverride::AsString => "as_string",
            LeafOverride::Decimal { .. } => "decimal(...)",
            LeafOverride::TimeUnit(_) => "time_unit = \"...\"",
        },
    };
    Err(syn::Error::new_spanned(
        field,
        format!(
            "field `{field_display_name}` has `{attr}` but its type is a tuple; \
             field-level attributes do not apply to multi-column tuple fields. \
             Hoist the tuple into a named struct that derives \
             `ToDataFrame` if you need per-element attributes."
        ),
    ))
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
fn unannotated_cow_bytes_error(field_display_name: &str, can_add_as_binary: bool) -> String {
    if can_add_as_binary {
        format!(
            "field `{field_display_name}` uses `Cow<'_, [u8]>` without `as_binary`; \
             add `#[df_derive(as_binary)]` to encode it as Binary, or use \
             `Vec<u8>` if you want the default `List(UInt8)` representation"
        )
    } else {
        format!(
            "field `{field_display_name}` contains `Cow<'_, [u8]>` in a tuple element; \
             tuple elements cannot be annotated with `as_binary`, so use `Vec<u8>` for \
             the default `List(UInt8)` representation or hoist the bytes into a named \
             struct field with `#[df_derive(as_binary)]`"
        )
    }
}

fn unannotated_borrowed_bytes_error(field_display_name: &str, can_add_as_binary: bool) -> String {
    if can_add_as_binary {
        format!(
            "field `{field_display_name}` uses `&[u8]` without `as_binary`; \
             add `#[df_derive(as_binary)]` to encode it as Binary, or use \
             `Vec<u8>` if you want the default `List(UInt8)` representation"
        )
    } else {
        format!(
            "field `{field_display_name}` contains `&[u8]` in a tuple element; \
             tuple elements cannot be annotated with `as_binary`, so use `Vec<u8>` \
             for the default `List(UInt8)` representation or hoist the bytes into \
             a named struct field with `#[df_derive(as_binary)]`"
        )
    }
}

fn borrowed_slice_error(field_display_name: &str) -> String {
    format!(
        "field `{field_display_name}` uses `&[T]`, but df-derive only \
         supports `&[u8]` with `#[df_derive(as_binary)]`; use `Vec<T>` \
         for list columns"
    )
}

fn cow_slice_error(field_display_name: &str) -> String {
    format!(
        "field `{field_display_name}` uses `Cow<'_, [T]>`, but df-derive only \
         supports `Cow<'_, [u8]>` with `#[df_derive(as_binary)]`; use `Vec<T>` \
         for list columns"
    )
}

fn default_leaf_for_base<S: ToTokens + ?Sized>(
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
        AnalyzedBase::BorrowedBytes => Err(syn::Error::new_spanned(
            span,
            unannotated_borrowed_bytes_error(field_display_name, can_add_as_binary),
        )),
        AnalyzedBase::CowBytes => Err(syn::Error::new_spanned(
            span,
            unannotated_cow_bytes_error(field_display_name, can_add_as_binary),
        )),
        AnalyzedBase::BorrowedSlice => Err(syn::Error::new_spanned(
            span,
            borrowed_slice_error(field_display_name),
        )),
        AnalyzedBase::CowSlice => Err(syn::Error::new_spanned(
            span,
            cow_slice_error(field_display_name),
        )),
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

/// Lower one analyzed tuple element to its IR form. Recurses into nested
/// tuples (`((i32, String), bool)`), preserves outer smart-pointer counts on
/// the element, and normalizes the element's wrapper stack independently of
/// the parent's. Field-level attributes are not applied here — they are
/// rejected on the parent field by [`reject_attrs_on_tuple`] before this runs.
fn analyzed_to_tuple_element(
    analyzed: AnalyzedType,
    field_display_name: &str,
) -> Result<TupleElement, syn::Error> {
    let leaf_spec =
        default_leaf_for_base(&analyzed.field_ty, field_display_name, analyzed.base, false)?;
    let wrapper_shape = normalize_wrappers(&analyzed.wrappers);
    Ok(TupleElement {
        leaf_spec,
        wrapper_shape,
        outer_smart_ptr_depth: analyzed.outer_smart_ptr_depth,
    })
}

const fn has_semantic_wrappers(wrappers: &[RawWrapper]) -> bool {
    !wrappers.is_empty()
}

fn reject_unsupported_wrapped_nested_tuples(
    analyzed: &AnalyzedType,
    field_display_name: &str,
) -> Result<(), syn::Error> {
    let AnalyzedBase::Tuple(elements) = &analyzed.base else {
        return Ok(());
    };
    let parent_wrapped = has_semantic_wrappers(&analyzed.wrappers);

    for element in elements {
        if matches!(element.base, AnalyzedBase::Tuple(_))
            && (parent_wrapped || has_semantic_wrappers(&element.wrappers))
        {
            return Err(syn::Error::new_spanned(
                &element.field_ty,
                format!(
                    "field `{field_display_name}` contains a nested tuple whose projection path \
                     is wrapped; nested tuples are supported only when each tuple on that path is \
                     unwrapped. Hoist the inner tuple into a named struct deriving `ToDataFrame`, \
                     or remove the `Option`/`Vec` wrapper around the tuple."
                ),
            ));
        }

        reject_unsupported_wrapped_nested_tuples(element, field_display_name)?;
    }

    Ok(())
}

fn is_direct_self_type(ty: &syn::Type, struct_name: &Ident) -> bool {
    let syn::Type::Path(type_path) = ty else {
        return false;
    };
    if type_path.qself.is_some() {
        return false;
    }
    let segments = &type_path.path.segments;
    let Some(segment) = segments.last() else {
        return false;
    };
    if segments.len() == 1 {
        return segment.ident == "Self" || &segment.ident == struct_name;
    }
    if segments.len() != 2
        || !segments
            .iter()
            .all(|segment| matches!(segment.arguments, syn::PathArguments::None))
    {
        return false;
    }
    let Some(first_segment) = segments.first() else {
        return false;
    };
    // Keep this intentionally narrow: broader qualified paths can name
    // distinct same-named types outside the deriving type's module.
    (first_segment.ident == "crate" || first_segment.ident == "self")
        && &segment.ident == struct_name
}

fn reject_direct_self_reference(
    analyzed: &AnalyzedType,
    field_display_name: &str,
    struct_name: &Ident,
) -> Result<(), syn::Error> {
    match &analyzed.base {
        AnalyzedBase::Struct(ty) if is_direct_self_type(ty, struct_name) => {
            Err(syn::Error::new_spanned(
                ty,
                format!(
                    "field `{field_display_name}` recursively references `{struct_name}` after \
                     transparent wrapper peeling; recursive nested DataFrame schemas are not \
                     supported. Store an identifier or foreign key, flatten the structure \
                     before deriving, or mark the field `#[df_derive(skip)]`."
                ),
            ))
        }
        AnalyzedBase::Tuple(elements) => {
            for element in elements {
                reject_direct_self_reference(element, field_display_name, struct_name)?;
            }
            Ok(())
        }
        _ => Ok(()),
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
        | AnalyzedBase::Decimal => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `as_str` but its base type does not implement \
                 `AsRef<str>`; `as_str` only applies to `String`, `&str`, `Cow<'_, str>`, \
                 custom struct types, or generic type parameters — drop the attribute or \
                 change the field type"
            ),
        )),
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
        AnalyzedBase::StdDuration => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `as_string`, but \
                 `std::time::Duration` and `core::time::Duration` do not implement \
                 `Display`; drop `as_string` to encode a Duration column, or wrap \
                 the value in a custom type that implements `Display`"
            ),
        )),
        AnalyzedBase::BorrowedBytes | AnalyzedBase::CowBytes => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `as_string`, but byte slices \
                 (`&[u8]`/`Cow<'_, [u8]>`) do not implement `Display`; use \
                 `#[df_derive(as_binary)]` for a Binary column, use `Vec<u8>` \
                 for a `List(UInt8)` column, or wrap the value in a custom type \
                 that implements `Display`"
            ),
        )),
        AnalyzedBase::BorrowedSlice | AnalyzedBase::CowSlice => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `as_string`, but borrowed slices \
                 (`&[T]`/`Cow<'_, [T]>`) do not implement `Display`; use `Vec<T>` \
                 for list columns, or wrap the value in a custom type that \
                 implements `Display`"
            ),
        )),
        // Tuple bases are rejected by `reject_attrs_on_tuple` before this
        // function runs. Keep this branch explicit so any bypass still emits
        // the same public diagnostic family instead of silently accepting it.
        AnalyzedBase::Tuple(_) => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `as_string` but its type is a tuple; \
                 field-level attributes do not apply to multi-column tuple fields. \
                 Hoist the tuple into a named struct that derives `ToDataFrame` if \
                 you need per-element attributes."
            ),
        )),
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
        _ => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `decimal(...)` but its base type is not \
                 a decimal backend candidate; `decimal(...)` applies to types named \
                 `Decimal`, custom struct types, or generic type parameters that \
                 implement `Decimal128Encode`"
            ),
        )),
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
        AnalyzedBase::NaiveDate => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
                 `chrono::NaiveDate`, which has a fixed encoding (i32 days since 1970-01-01) \
                 and offers no unit choice — remove the attribute"
            ),
        )),
        AnalyzedBase::NaiveTime => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
                 `chrono::NaiveTime`, which has a fixed encoding (i64 nanoseconds since \
                 midnight) and offers no unit choice — remove the attribute"
            ),
        )),
        _ => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
                 not `chrono::DateTime<Tz>`, `chrono::NaiveDateTime`, \
                 `std::time::Duration`, `core::time::Duration`, or \
                 `chrono::Duration`; remove the attribute or change the field type"
            ),
        )),
    }
}

/// Normalize the raw outer-to-inner `RawWrapper` sequence into a
/// `WrapperShape` the encoder consumes directly. `Option` and smart-pointer
/// steps are retained as an `AccessChain` at each wrapper boundary: above
/// each `Vec`, immediately surrounding the leaf, or for the leaf-only path.
/// Polars folds consecutive `Option`s into a single validity bit per
/// position, so the count is also cached to choose the direct single-Option
/// path versus the collapsed multi-Option path.
fn normalize_wrappers(wrappers: &[RawWrapper]) -> WrapperShape {
    let mut layers: Vec<VecLayerSpec> = Vec::new();
    let mut pending_access = AccessChain::empty();
    for w in wrappers {
        match w {
            RawWrapper::Option => {
                pending_access.steps.push(AccessStep::Option);
            }
            RawWrapper::SmartPtr => {
                pending_access.steps.push(AccessStep::SmartPtr);
            }
            RawWrapper::Vec => {
                let option_layers_above = pending_access.option_layers();
                layers.push(VecLayerSpec {
                    option_layers_above,
                    access: std::mem::take(&mut pending_access),
                });
            }
        }
    }
    let Some(layers) = NonEmpty::from_vec(layers) else {
        return WrapperShape::Leaf(LeafShape::from_option_access(
            pending_access.option_layers(),
            pending_access,
        ));
    };
    WrapperShape::Vec(VecLayers {
        layers,
        inner_option_layers: pending_access.option_layers(),
        inner_access: pending_access,
    })
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
    let bare_u8_msg = || {
        format!(
            "field `{field_display_name}` has `as_binary` but its type is a single `u8`; \
             `as_binary` requires a `Vec<u8>` shape — bare `u8` is a single byte, not \
             a binary blob. Wrap the field in `Vec<u8>` to opt into Binary."
        )
    };
    let inner_option_msg = || {
        format!(
            "field `{field_display_name}` has `as_binary` but the wrapper stack places an \
             `Option` between the `Vec` and the `u8` leaf; BinaryView cannot carry \
             per-byte nulls. Use `Vec<u8>` directly (drop the inner `Option`)."
        )
    };
    let wrong_base_msg = || {
        format!(
            "field `{field_display_name}` has `as_binary` but its base type is not `u8`; \
             `as_binary` requires a `Vec<u8>` shape (the innermost `Vec` becomes the \
             Binary blob). Change the field type or drop the attribute."
        )
    };
    let cow_slice_msg = || {
        format!(
            "field `{field_display_name}` has `as_binary` on `Cow<'_, [T]>`, but \
             `as_binary` only supports `Cow<'_, [u8]>`; use `Vec<T>` for list columns"
        )
    };
    let borrowed_slice_msg = || {
        format!(
            "field `{field_display_name}` has `as_binary` on `&[T]`, but \
             `as_binary` only supports `&[u8]`; use `Vec<T>` for list columns"
        )
    };
    if matches!(base, AnalyzedBase::CowBytes | AnalyzedBase::BorrowedBytes) {
        return Ok((LeafSpec::Binary, wrappers.to_vec()));
    }
    if matches!(base, AnalyzedBase::BorrowedSlice) {
        return Err(syn::Error::new_spanned(field, borrowed_slice_msg()));
    }
    if matches!(base, AnalyzedBase::CowSlice) {
        return Err(syn::Error::new_spanned(field, cow_slice_msg()));
    }
    if !matches!(base, AnalyzedBase::Numeric(NumericKind::U8)) {
        return Err(syn::Error::new_spanned(field, wrong_base_msg()));
    }
    match wrappers.last() {
        None => Err(syn::Error::new_spanned(field, bare_u8_msg())),
        Some(RawWrapper::Option) => {
            // Either bare `Option<u8>` (single `Option` wrapper) or any deeper
            // stack ending in `Option`-immediately-above-`u8`. Both share the
            // "no per-byte nulls" rejection.
            if wrappers.len() == 1 {
                Err(syn::Error::new_spanned(field, bare_u8_msg()))
            } else {
                Err(syn::Error::new_spanned(field, inner_option_msg()))
            }
        }
        Some(RawWrapper::Vec) => {
            let mut trimmed = wrappers.to_vec();
            trimmed.pop();
            Ok((LeafSpec::Binary, trimmed))
        }
        Some(RawWrapper::SmartPtr) => Err(syn::Error::new_spanned(field, wrong_base_msg())),
    }
}

/// Run the per-field pipeline (override parsing, type analysis, leaf-spec
/// resolution, wrapper normalization) and produce the corresponding `FieldIR`.
/// Named and tuple arms share this body; they only differ in how `name_ident`
/// and `field_index` are derived from the surrounding `Fields` shape.
fn process_field(
    field: &syn::Field,
    name_ident: Ident,
    field_index: Option<usize>,
    struct_name: &Ident,
    generic_params: &[Ident],
) -> Result<Option<FieldIR>, syn::Error> {
    let display_name = name_ident.to_string();
    let override_ = parse_field_override(field, &display_name)?;
    if matches!(override_, Some(FieldOverride::Skip)) {
        return Ok(None);
    }
    let analyzed = analyze_type(&field.ty, generic_params)?;
    reject_direct_self_reference(&analyzed, &display_name, struct_name)?;
    reject_unsupported_wrapped_nested_tuples(&analyzed, &display_name)?;
    let outer_smart_ptr_depth = analyzed.outer_smart_ptr_depth;
    let decimal_generic_params =
        decimal_generic_params_for_override(override_.as_ref(), &analyzed.base);
    let decimal_backend_ty = decimal_backend_ty_for_override(override_.as_ref(), &analyzed.base);
    let (leaf_spec, wrapper_shape) = if matches!(override_, Some(FieldOverride::AsBinary)) {
        // `as_binary` over a tuple is rejected here too — `parse_as_binary_shape`
        // only checks the leaf base, but the tuple itself fails the same
        // multi-column attribute rule as every other field-level attribute.
        if matches!(analyzed.base, AnalyzedBase::Tuple(_)) {
            reject_attrs_on_tuple(
                field,
                &display_name,
                override_.as_ref().map(FieldOverrideRef::Field),
            )?;
        }
        let (leaf, trimmed) =
            parse_as_binary_shape(field, &display_name, &analyzed.base, &analyzed.wrappers)?;
        (leaf, normalize_wrappers(&trimmed))
    } else {
        let leaf = parse_leaf_spec(
            field,
            &display_name,
            override_.as_ref().and_then(FieldOverride::leaf),
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

/// Parse a `syn::DeriveInput` into the IR consumed by codegen.
///
/// Returns a `syn::Error` for non-struct inputs (enums, unions). Tuple structs
/// and unit structs are supported.
pub fn parse_to_ir(input: &DeriveInput) -> Result<StructIR, syn::Error> {
    let name = input.ident.clone();
    let generics = input.generics.clone();
    let generic_params: Vec<Ident> = generics.type_params().map(|tp| tp.ident.clone()).collect();
    let mut fields_ir: Vec<FieldIR> = Vec::new();

    let data_struct = match &input.data {
        Data::Struct(data_struct) => data_struct,
        Data::Enum(data_enum) => {
            return Err(syn::Error::new(
                data_enum.enum_token.span,
                "df-derive cannot be derived on enums; derive `ToDataFrame` on a struct \
                 and use `#[df_derive(as_string)]` on enum fields",
            ));
        }
        Data::Union(data_union) => {
            return Err(syn::Error::new(
                data_union.union_token.span,
                "df-derive cannot be derived on unions; derive `ToDataFrame` on a struct",
            ));
        }
    };

    match &data_struct.fields {
        Fields::Named(named) => {
            for field in &named.named {
                let name_ident = field
                    .ident
                    .as_ref()
                    .expect("named fields must have ident")
                    .clone();
                if let Some(field_ir) =
                    process_field(field, name_ident, None, &name, &generic_params)?
                {
                    fields_ir.push(field_ir);
                }
            }
        }
        Fields::Unit => {}
        Fields::Unnamed(unnamed) => {
            for (index, field) in unnamed.unnamed.iter().enumerate() {
                let name_ident = format_ident!("field_{}", index);
                if let Some(field_ir) =
                    process_field(field, name_ident, Some(index), &name, &generic_params)?
                {
                    fields_ir.push(field_ir);
                }
            }
        }
    }

    Ok(StructIR {
        name,
        generics,
        fields: fields_ir,
    })
}
