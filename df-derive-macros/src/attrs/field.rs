use crate::ir::DateTimeUnit;
use proc_macro2::Span;
use syn::spanned::Spanned as SynSpanned;

use super::Spanned;
use super::decimal::parse_decimal_attr;
use super::field_conflicts::{FieldAttr, set_override};

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
                        field_display_name,
                        &mut override_,
                        FieldAttr::Skip,
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_string") {
                    set_override(
                        field_display_name,
                        &mut override_,
                        FieldAttr::Leaf(LeafOverride::AsString),
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_str") {
                    set_override(
                        field_display_name,
                        &mut override_,
                        FieldAttr::Leaf(LeafOverride::AsStr),
                        incoming_span,
                    )
                } else if meta.path.is_ident("as_binary") {
                    set_override(
                        field_display_name,
                        &mut override_,
                        FieldAttr::Binary,
                        incoming_span,
                    )
                } else if meta.path.is_ident("decimal") {
                    let (precision, scale) = parse_decimal_attr(&meta)?;
                    set_override(
                        field_display_name,
                        &mut override_,
                        FieldAttr::Leaf(LeafOverride::Decimal { precision, scale }),
                        incoming_span,
                    )
                } else if meta.path.is_ident("time_unit") {
                    let unit = parse_time_unit_attr(&meta)?;
                    set_override(
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

    #[test]
    fn conflicting_overrides_report_the_first_override() {
        let error = parse_disposition(&syn::parse_quote! {
            #[df_derive(as_str, as_string)]
            value: String
        })
        .expect_err("conflicting field overrides should fail");
        let rendered = error.into_compile_error().to_string();

        assert!(rendered.contains("has both `as_str` and `as_string`"));
        assert!(rendered.contains("first `as_str` override declared here"));
    }
}
