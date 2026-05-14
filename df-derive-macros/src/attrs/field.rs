use crate::ir::DateTimeUnit;
use proc_macro2::Span;
use syn::spanned::Spanned;

/// Leaf-level conversion override declared via `#[df_derive(...)]`.
/// `skip` and `as_binary` are field dispositions, so they are deliberately
/// not representable here and cannot reach leaf parsing.
pub enum LeafOverride {
    AsStr,
    AsString,
    Decimal { precision: u8, scale: u8 },
    TimeUnit(DateTimeUnit),
}

/// Mutually-exclusive field-level override declared via `#[df_derive(...)]`.
pub enum FieldOverride {
    Skip,
    AsBinary,
    Leaf(LeafOverride),
}

impl FieldOverride {
    pub const fn leaf(&self) -> Option<&LeafOverride> {
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

pub fn parse_field_override(
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
