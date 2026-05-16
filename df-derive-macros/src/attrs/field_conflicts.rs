use crate::ir::DateTimeUnit;
use proc_macro2::Span;

use super::Spanned;
use super::field::{FieldConversion, FieldDisposition, LeafOverride};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FieldAttr {
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

    pub(super) const fn into_disposition(self, span: Span) -> FieldDisposition {
        match self {
            Self::Skip => FieldDisposition::Skip,
            Self::Binary => FieldDisposition::Include(FieldConversion::Binary { span }),
            Self::Leaf(leaf_override) => {
                FieldDisposition::Include(FieldConversion::LeafOverride(Spanned {
                    value: leaf_override,
                    span,
                }))
            }
        }
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

fn conflict_message(field_display_name: &str, existing: FieldAttr, incoming: FieldAttr) -> String {
    use FieldOverrideKey::{AsBinary, AsStr, AsString, Decimal, Skip, TimeUnit};

    match (existing.key(), incoming.key()) {
        (AsStr, AsString) | (AsString, AsStr) => {
            format!(
                "field `{field_display_name}` has both `as_str` and `as_string`; \
                 pick one — `as_str` borrows via `AsRef<str>` without formatting, \
                 `as_string` formats via `Display` into a reused scratch buffer"
            )
        }
        (Decimal, AsStr | AsString) | (AsStr | AsString, Decimal) => {
            format!(
                "field `{field_display_name}` combines `decimal(...)` with `as_str`/`as_string`; \
                 `as_str`/`as_string` produce a String column, so the `decimal(...)` \
                 dtype override has no effect — drop one"
            )
        }
        (TimeUnit, AsStr | AsString) | (AsStr | AsString, TimeUnit) => {
            format!(
                "field `{field_display_name}` combines `time_unit = \"...\"` with \
                 `as_str`/`as_string`; the latter produces a String column, so the \
                 `time_unit` override has no effect — drop one"
            )
        }
        (Decimal, TimeUnit) | (TimeUnit, Decimal) => format!(
            "field `{field_display_name}` combines `decimal(...)` with `time_unit = \"...\"`; \
             pick one — `decimal(...)` applies to decimal backend candidates, \
             `time_unit` only applies to `chrono::DateTime<Tz>`, \
             `chrono::NaiveDateTime`, `std::time::Duration`, \
             `core::time::Duration`, or `chrono::Duration`"
        ),
        (Skip, _) | (_, Skip) => format!(
            "field `{field_display_name}` combines `skip` with another field attribute; \
             `skip` omits the field entirely, so conversion attributes have no effect; drop one"
        ),
        (AsBinary, _) | (_, AsBinary) => format!(
            "field `{field_display_name}` combines `as_binary` with another override; \
             `as_binary` produces a Binary column over a `Vec<u8>` shape and is \
             mutually exclusive with `as_str`, `as_string`, `decimal(...)`, and \
             `time_unit = \"...\"` — drop one"
        ),
        _ => {
            format!("field `{field_display_name}` combines incompatible field attributes; drop one")
        }
    }
}

fn override_conflict(
    field_display_name: &str,
    existing: FieldAttr,
    existing_span: Span,
    incoming: FieldAttr,
    incoming_span: Span,
) -> syn::Error {
    let mut error = syn::Error::new(
        incoming_span,
        conflict_message(field_display_name, existing, incoming),
    );
    error.combine(syn::Error::new(
        existing_span,
        format!("first `{}` override declared here", existing.label()),
    ));
    error
}

pub(super) fn set_override(
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
        Some((existing, existing_span)) if existing.key() == incoming.key() => {
            Err(duplicate_override_conflict(
                field_display_name,
                *existing,
                incoming,
                *existing_span,
                incoming_span,
            ))
        }
        Some((existing, existing_span)) => Err(override_conflict(
            field_display_name,
            *existing,
            *existing_span,
            incoming,
            incoming_span,
        )),
    }
}
