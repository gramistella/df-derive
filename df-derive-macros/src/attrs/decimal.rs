use proc_macro2::Span;
use syn::spanned::Spanned as SynSpanned;

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

pub(super) fn parse_decimal_attr(
    meta: &syn::meta::ParseNestedMeta<'_>,
) -> Result<(u8, u8), syn::Error> {
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
