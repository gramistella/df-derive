use crate::ir::DateTimeUnit;
use proc_macro2::Span;
use syn::spanned::Spanned as SynSpanned;

use super::Spanned;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafOverride {
    AsStr,
    AsString,
    Decimal { precision: u8, scale: u8 },
    TimeUnit(DateTimeUnit),
}

#[derive(Clone, Debug)]
pub enum FieldDisposition {
    Include {
        leaf_override: Option<Spanned<LeafOverride>>,
    },
    Skip,
    Binary {
        span: Span,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldAttr {
    Skip,
    Binary,
    Leaf(LeafOverride),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldOverrideKey {
    Skip,
    AsBinary,
    AsStr,
    AsString,
    Decimal,
    TimeUnit,
}

impl FieldAttr {
    const fn key(self) -> FieldOverrideKey {
        match self {
            Self::Skip => FieldOverrideKey::Skip,
            Self::Binary => FieldOverrideKey::AsBinary,
            Self::Leaf(LeafOverride::AsStr) => FieldOverrideKey::AsStr,
            Self::Leaf(LeafOverride::AsString) => FieldOverrideKey::AsString,
            Self::Leaf(LeafOverride::Decimal { .. }) => FieldOverrideKey::Decimal,
            Self::Leaf(LeafOverride::TimeUnit(_)) => FieldOverrideKey::TimeUnit,
        }
    }

    const fn label(self) -> &'static str {
        match self.key() {
            FieldOverrideKey::Skip => "skip",
            FieldOverrideKey::AsBinary => "as_binary",
            FieldOverrideKey::AsStr => "as_str",
            FieldOverrideKey::AsString => "as_string",
            FieldOverrideKey::Decimal => "decimal(...)",
            FieldOverrideKey::TimeUnit => "time_unit",
        }
    }

    const fn into_disposition(self, span: Span) -> FieldDisposition {
        match self {
            Self::Skip => FieldDisposition::Skip,
            Self::Binary => FieldDisposition::Binary { span },
            Self::Leaf(leaf_override) => FieldDisposition::Include {
                leaf_override: Some(Spanned {
                    value: leaf_override,
                    span,
                }),
            },
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

fn override_conflict(
    field: &syn::Field,
    field_display_name: &str,
    existing: FieldAttr,
    incoming: FieldAttr,
) -> syn::Error {
    let message = match (existing, incoming) {
        (FieldAttr::Leaf(LeafOverride::AsStr), FieldAttr::Leaf(LeafOverride::AsString))
        | (FieldAttr::Leaf(LeafOverride::AsString), FieldAttr::Leaf(LeafOverride::AsStr)) => {
            format!(
                "field `{field_display_name}` has both `as_str` and `as_string`; \
             pick one — `as_str` borrows via `AsRef<str>` without formatting, \
             `as_string` formats via `Display` into a reused scratch buffer"
            )
        }
        (
            FieldAttr::Leaf(LeafOverride::Decimal { .. }),
            FieldAttr::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
        )
        | (
            FieldAttr::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
            FieldAttr::Leaf(LeafOverride::Decimal { .. }),
        ) => {
            format!(
                "field `{field_display_name}` combines `decimal(...)` with `as_str`/`as_string`; \
                 `as_str`/`as_string` produce a String column, so the `decimal(...)` \
                 dtype override has no effect — drop one"
            )
        }
        (
            FieldAttr::Leaf(LeafOverride::TimeUnit(_)),
            FieldAttr::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
        )
        | (
            FieldAttr::Leaf(LeafOverride::AsStr | LeafOverride::AsString),
            FieldAttr::Leaf(LeafOverride::TimeUnit(_)),
        ) => {
            format!(
                "field `{field_display_name}` combines `time_unit = \"...\"` with \
                 `as_str`/`as_string`; the latter produces a String column, so the \
                 `time_unit` override has no effect — drop one"
            )
        }
        (
            FieldAttr::Leaf(LeafOverride::Decimal { .. }),
            FieldAttr::Leaf(LeafOverride::TimeUnit(_)),
        )
        | (
            FieldAttr::Leaf(LeafOverride::TimeUnit(_)),
            FieldAttr::Leaf(LeafOverride::Decimal { .. }),
        ) => format!(
            "field `{field_display_name}` combines `decimal(...)` with `time_unit = \"...\"`; \
             pick one — `decimal(...)` applies to decimal backend candidates, \
             `time_unit` only applies to `chrono::DateTime<Tz>`, \
             `chrono::NaiveDateTime`, `std::time::Duration`, \
             `core::time::Duration`, or `chrono::Duration`"
        ),
        (FieldAttr::Skip, _) | (_, FieldAttr::Skip) => format!(
            "field `{field_display_name}` combines `skip` with another field attribute; \
             `skip` omits the field entirely, so conversion attributes have no effect; drop one"
        ),
        (FieldAttr::Binary, _) | (_, FieldAttr::Binary) => format!(
            "field `{field_display_name}` combines `as_binary` with another override; \
             `as_binary` produces a Binary column over a `Vec<u8>` shape and is \
             mutually exclusive with `as_str`, `as_string`, `decimal(...)`, and \
             `time_unit = \"...\"` — drop one"
        ),
        _ if existing.key() == incoming.key() => format!(
            "field `{field_display_name}` declares duplicate `{}` override; remove one",
            incoming.label()
        ),
        _ => {
            format!("field `{field_display_name}` combines incompatible field attributes; drop one")
        }
    };
    syn::Error::new_spanned(field, message)
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
    existing: FieldAttr,
    incoming: FieldAttr,
    existing_span: Span,
    incoming_span: Span,
) -> syn::Error {
    let key = incoming.label();
    let message = match (existing, incoming) {
        (
            FieldAttr::Leaf(LeafOverride::Decimal {
                precision: existing_precision,
                scale: existing_scale,
            }),
            FieldAttr::Leaf(LeafOverride::Decimal {
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
            FieldAttr::Leaf(LeafOverride::TimeUnit(existing_unit)),
            FieldAttr::Leaf(LeafOverride::TimeUnit(incoming_unit)),
        ) if existing_unit != incoming_unit => {
            let existing_unit = time_unit_attr_value(existing_unit);
            let incoming_unit = time_unit_attr_value(incoming_unit);
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

fn set_override(
    field: &syn::Field,
    field_display_name: &str,
    override_: &mut Option<(FieldAttr, Span)>,
    incoming: FieldAttr,
    incoming_span: Span,
) -> Result<(), syn::Error> {
    match override_ {
        None => {
            *override_ = Some((incoming, incoming_span));
            Ok(())
        }
        Some((existing, existing_span)) if same_override_key(*existing, incoming) => {
            Err(duplicate_override_conflict(
                field_display_name,
                *existing,
                incoming,
                *existing_span,
                incoming_span,
            ))
        }
        Some((existing, _)) => Err(override_conflict(
            field,
            field_display_name,
            *existing,
            incoming,
        )),
    }
}

fn same_override_key(existing: FieldAttr, incoming: FieldAttr) -> bool {
    existing.key() == incoming.key()
}

pub fn parse_field_disposition(
    field: &syn::Field,
    field_display_name: &str,
) -> Result<FieldDisposition, syn::Error> {
    let mut override_: Option<(FieldAttr, Span)> = None;
    for attr in &field.attrs {
        if attr.path().is_ident("df_derive") {
            attr.parse_nested_meta(|meta| {
                let incoming_span = meta.path.span();
                if meta.path.is_ident("skip") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldAttr::Skip,
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_string") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldAttr::Leaf(LeafOverride::AsString),
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_str") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldAttr::Leaf(LeafOverride::AsStr),
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_binary") {
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldAttr::Binary,
                        incoming_span,
                    )
                } else if meta.path.is_ident("decimal") {
                    let (precision, scale) = parse_decimal_attr(&meta)?;
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldAttr::Leaf(LeafOverride::Decimal { precision, scale }),
                        incoming_span,
                    )
                } else if meta.path.is_ident("time_unit") {
                    let unit = parse_time_unit_attr(&meta)?;
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldAttr::Leaf(LeafOverride::TimeUnit(unit)),
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
    Ok(override_.map_or(
        FieldDisposition::Include {
            leaf_override: None,
        },
        |(value, span)| value.into_disposition(span),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_disposition(field: &syn::Field) -> syn::Result<FieldDisposition> {
        parse_field_disposition(field, "value")
    }

    fn leaf_override_value(field: &syn::Field) -> LeafOverride {
        let disposition = parse_disposition(field).expect("field disposition should parse");
        let FieldDisposition::Include {
            leaf_override: Some(leaf_override),
        } = disposition
        else {
            panic!("field leaf override should be present");
        };
        leaf_override.value
    }

    #[test]
    fn parses_string_and_decimal_field_overrides() {
        let as_str = leaf_override_value(&syn::parse_quote! {
            #[df_derive(as_str)]
            value: String
        });
        assert!(matches!(as_str, LeafOverride::AsStr));

        let as_string = leaf_override_value(&syn::parse_quote! {
            #[df_derive(as_string)]
            value: DisplayType
        });
        assert!(matches!(as_string, LeafOverride::AsString));

        let decimal = leaf_override_value(&syn::parse_quote! {
            #[df_derive(decimal(precision = 10, scale = 2))]
            value: Decimal
        });
        assert!(matches!(
            decimal,
            LeafOverride::Decimal {
                precision: 10,
                scale: 2,
            }
        ));
    }

    #[test]
    fn rejects_duplicate_decimal_keys_and_bad_time_units() {
        let duplicate_decimal = parse_disposition(&syn::parse_quote! {
            #[df_derive(decimal(precision = 10, precision = 11, scale = 2))]
            value: Decimal
        });
        assert!(duplicate_decimal.is_err());

        let bad_time_unit = parse_disposition(&syn::parse_quote! {
            #[df_derive(time_unit = "bad")]
            value: chrono::NaiveDateTime
        });
        assert!(bad_time_unit.is_err());
    }

    #[test]
    fn rejects_conflicting_string_overrides() {
        let result = parse_disposition(&syn::parse_quote! {
            #[df_derive(as_str, as_string)]
            value: String
        });
        assert!(result.is_err());
    }
}
