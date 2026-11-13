//! Parsing and validation of `#[df_derive(...)]` attributes.

use proc_macro2::Span;
use syn::spanned::Spanned as _;
use syn::{Attribute, LitInt, LitStr};

/// Attributes on the derived type.
pub struct ContainerAttrs {
    /// Path of the crate that exposes the runtime as `<path>::dataframe`.
    pub krate: syn::Path,
    /// The type is a single-field wrapper that encodes exactly as its field.
    pub transparent: Option<Span>,
    /// Encode through another row type built from a reference to this one.
    pub via: Option<Via>,
    /// Extra where-clause predicates for the generated impls.
    pub bounds: Vec<syn::WherePredicate>,
}

/// A projection chosen with `via = "..."` or `try_via = "..."`.
pub struct Via {
    /// The row type, written with `'_` for lifetimes borrowed from the value.
    pub row: syn::Type,
    /// Whether the conversion is `TryFrom` rather than `From`.
    pub fallible: bool,
    pub span: Span,
}

/// A time unit named by `time_unit = "..."`.
#[derive(Clone, Copy)]
pub enum TimeUnit {
    Ms,
    Us,
    Ns,
}

/// The encoding chosen for a field's innermost value.
pub enum Encoding {
    AsStr,
    AsString,
    AsBinary,
    /// A categorical column of the text the value borrows as, or with
    /// `display` of the text it formats as.
    Categorical {
        display: bool,
    },
    Decimal {
        precision: u8,
        scale: u8,
    },
    TimeUnit(TimeUnit),
    With(Box<syn::Type>),
}

impl Encoding {
    const fn attribute(&self) -> &'static str {
        match self {
            Self::AsStr => "as_str",
            Self::AsString => "as_string",
            Self::AsBinary => "as_binary",
            Self::Categorical { .. } => "categorical",
            Self::Decimal { .. } => "decimal",
            Self::TimeUnit(_) => "time_unit",
            Self::With(_) => "with",
        }
    }
}

/// Where a flattened field's columns go.
pub enum Flatten {
    /// Directly into the parent, with no prefix.
    Bare,
    /// Under an explicit prefix instead of the field name.
    Prefix(LitStr),
}

/// Attributes on one field.
#[derive(Default)]
pub struct FieldAttrs {
    pub skip: Option<Span>,
    /// The name the field's columns take instead of the field's own.
    pub rename: Option<(LitStr, Span)>,
    pub flatten: Option<(Flatten, Span)>,
    pub encoding: Option<(Encoding, Span)>,
}

fn df_derive_attrs(attrs: &[Attribute]) -> impl Iterator<Item = &Attribute> {
    attrs
        .iter()
        .filter(|attr| attr.path().is_ident("df_derive"))
}

pub fn container(attrs: &[Attribute]) -> syn::Result<ContainerAttrs> {
    let mut krate: Option<syn::Path> = None;
    let mut transparent: Option<Span> = None;
    let mut via: Option<Via> = None;
    let mut bounds = Vec::new();
    for attr in df_derive_attrs(attrs) {
        attr.parse_nested_meta(|meta| {
            let fallible = meta.path.is_ident("try_via");
            if fallible || meta.path.is_ident("via") {
                if via.is_some() {
                    return Err(meta.error("a type has one `via` or `try_via` projection"));
                }
                let span = meta.path.span();
                let literal: LitStr = meta.value()?.parse()?;
                via = Some(Via {
                    row: literal.parse()?,
                    fallible,
                    span,
                });
                return Ok(());
            }
            if meta.path.is_ident("bound") {
                let literal: LitStr = meta.value()?.parse()?;
                let parsed = literal.parse_with(
                    syn::punctuated::Punctuated::<syn::WherePredicate, syn::Token![,]>::parse_terminated,
                )?;
                bounds.extend(parsed);
                return Ok(());
            }
            if meta.path.is_ident("transparent") {
                if transparent.is_some() {
                    return Err(meta.error("duplicate `transparent` attribute"));
                }
                transparent = Some(meta.path.span());
                return Ok(());
            }
            if meta.path.is_ident("crate") {
                if krate.is_some() {
                    return Err(meta.error("duplicate `crate` attribute"));
                }
                let literal: LitStr = meta.value()?.parse()?;
                krate = Some(literal.parse()?);
                return Ok(());
            }
            Err(meta.error(
                "unknown container attribute; expected one of `crate = \"...\"`, `transparent`, \
                 `via = \"...\"`, `try_via = \"...\"`, `bound = \"...\"`",
            ))
        })?;
    }
    if let (Some(span), Some(_)) = (transparent, &via) {
        return Err(syn::Error::new(
            span,
            "`transparent` cannot be combined with `via`: a projected type encodes as its row type",
        ));
    }
    Ok(ContainerAttrs {
        krate: krate.unwrap_or_else(|| syn::parse_quote!(::df_derive)),
        transparent,
        via,
        bounds,
    })
}

/// Rejects `#[df_derive(...)]` inside a type that encodes through `via`.
///
/// The columns of such a type are those of its row type. Its own fields and
/// variants are never read, so an attribute on one would do nothing, and
/// would not even be checked.
pub fn none_inside(data: &syn::Data) -> syn::Result<()> {
    fn of_fields(fields: &syn::Fields) -> impl Iterator<Item = &Attribute> {
        fields.iter().flat_map(|field| &field.attrs)
    }

    let attrs: Vec<&Attribute> = match data {
        syn::Data::Struct(data) => of_fields(&data.fields).collect(),
        syn::Data::Enum(data) => data
            .variants
            .iter()
            .flat_map(|variant| variant.attrs.iter().chain(of_fields(&variant.fields)))
            .collect(),
        syn::Data::Union(data) => data
            .fields
            .named
            .iter()
            .flat_map(|field| &field.attrs)
            .collect(),
    };
    attrs
        .into_iter()
        .find(|attr| attr.path().is_ident("df_derive"))
        .map_or(Ok(()), |attr| {
            Err(syn::Error::new_spanned(
                &attr.meta,
                "this attribute has no effect: a type with `via` encodes as its row type, \
                 so annotate that type's fields instead",
            ))
        })
}

fn decimal(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<Encoding> {
    let mut precision: Option<u8> = None;
    let mut scale: Option<u8> = None;
    meta.parse_nested_meta(|sub| {
        let slot = if sub.path.is_ident("precision") {
            &mut precision
        } else if sub.path.is_ident("scale") {
            &mut scale
        } else {
            return Err(sub.error("expected `precision = N` or `scale = N`"));
        };
        if slot.is_some() {
            return Err(sub.error("this key is given more than once"));
        }
        let literal: LitInt = sub.value()?.parse()?;
        let value: u8 = literal
            .base10_parse()
            .map_err(|_| syn::Error::new(literal.span(), "expected an integer between 0 and 38"))?;
        *slot = Some(value);
        Ok(())
    })?;
    let (Some(precision), Some(scale)) = (precision, scale) else {
        return Err(meta.error("`decimal` needs both `precision = N` and `scale = N`"));
    };
    if !(1..=38).contains(&precision) {
        return Err(meta.error(format!(
            "decimal precision must satisfy 1 <= precision <= 38 (got {precision})"
        )));
    }
    if scale > precision {
        return Err(meta.error(format!(
            "decimal scale must not exceed precision (got scale {scale}, precision {precision})"
        )));
    }
    Ok(Encoding::Decimal { precision, scale })
}

fn time_unit(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<Encoding> {
    let literal: LitStr = meta.value()?.parse()?;
    let unit = match literal.value().as_str() {
        "ms" => TimeUnit::Ms,
        "us" => TimeUnit::Us,
        "ns" => TimeUnit::Ns,
        other => {
            return Err(syn::Error::new(
                literal.span(),
                format!("unknown time unit `{other}`; expected \"ms\", \"us\" or \"ns\""),
            ));
        }
    };
    Ok(Encoding::TimeUnit(unit))
}

fn flatten(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<Flatten> {
    if !meta.input.peek(syn::token::Paren) {
        return Ok(Flatten::Bare);
    }
    let mut prefix: Option<LitStr> = None;
    meta.parse_nested_meta(|sub| {
        if !sub.path.is_ident("prefix") {
            return Err(sub.error("expected `prefix = \"...\"`"));
        }
        if prefix.is_some() {
            return Err(sub.error("`prefix` is given more than once"));
        }
        let literal: LitStr = sub.value()?.parse()?;
        if literal.value().is_empty() {
            return Err(syn::Error::new(
                literal.span(),
                "flatten prefix must not be empty; use plain `flatten` for no prefix",
            ));
        }
        prefix = Some(literal);
        Ok(())
    })?;
    prefix
        .map(Flatten::Prefix)
        .ok_or_else(|| meta.error("expected `flatten(prefix = \"...\")`"))
}

pub fn field(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
    let mut parsed = FieldAttrs::default();
    // `categorical` says how text is stored, and goes with either way of
    // reading it, so it is joined with the encoding once both are known.
    let mut categorical: Option<Span> = None;
    for attr in df_derive_attrs(attrs) {
        attr.parse_nested_meta(|meta| {
            let span = meta.path.span();
            if meta.path.is_ident("categorical") {
                if categorical.is_some() {
                    return Err(meta.error("duplicate `categorical` attribute"));
                }
                categorical = Some(span);
                return Ok(());
            }
            if meta.path.is_ident("skip") {
                if parsed.skip.is_some() {
                    return Err(meta.error("duplicate `skip` attribute"));
                }
                parsed.skip = Some(span);
                return Ok(());
            }
            if meta.path.is_ident("rename") {
                if parsed.rename.is_some() {
                    return Err(meta.error("duplicate `rename` attribute"));
                }
                let name: LitStr = meta.value()?.parse()?;
                if name.value().is_empty() {
                    return Err(syn::Error::new(
                        name.span(),
                        "a column name must not be empty",
                    ));
                }
                parsed.rename = Some((name, span));
                return Ok(());
            }
            if meta.path.is_ident("flatten") {
                if parsed.flatten.is_some() {
                    return Err(meta.error("duplicate `flatten` attribute"));
                }
                parsed.flatten = Some((flatten(&meta)?, span));
                return Ok(());
            }
            let encoding = if meta.path.is_ident("as_str") {
                Encoding::AsStr
            } else if meta.path.is_ident("as_string") {
                Encoding::AsString
            } else if meta.path.is_ident("as_binary") {
                Encoding::AsBinary
            } else if meta.path.is_ident("decimal") {
                decimal(&meta)?
            } else if meta.path.is_ident("time_unit") {
                time_unit(&meta)?
            } else if meta.path.is_ident("with") {
                let literal: LitStr = meta.value()?.parse()?;
                Encoding::With(Box::new(literal.parse()?))
            } else {
                return Err(meta.error(
                    "unknown field attribute; expected one of `skip`, `rename = \"...\"`, \
                     `flatten`, `as_str`, `as_string`, `as_binary`, `categorical`, \
                     `decimal(...)`, `time_unit = \"...\"`, `with = \"...\"`",
                ));
            };
            if let Some((existing, _)) = &parsed.encoding {
                return Err(meta.error(format!(
                    "`{}` cannot be combined with `{}`: a field has one encoding",
                    encoding.attribute(),
                    existing.attribute(),
                )));
            }
            parsed.encoding = Some((encoding, span));
            Ok(())
        })?;
    }

    if let Some(span) = categorical {
        parsed.encoding = Some((as_categories(parsed.encoding.as_ref(), span)?, span));
    }
    without_conflicts(parsed)
}

/// The encoding of a field marked `categorical`, given the encoding its
/// other attributes chose: the text is read as that one reads it.
fn as_categories(encoding: Option<&(Encoding, Span)>, span: Span) -> syn::Result<Encoding> {
    let display = match encoding {
        None | Some((Encoding::AsStr, _)) => false,
        Some((Encoding::AsString, _)) => true,
        Some((other, _)) => {
            return Err(syn::Error::new(
                span,
                format!(
                    "`categorical` cannot be combined with `{}`: a categorical column holds \
                     text, read as `as_str` reads it or, with `as_string`, as `Display` writes it",
                    other.attribute()
                ),
            ));
        }
    };
    Ok(Encoding::Categorical { display })
}

/// Rejects attributes that contradict each other.
fn without_conflicts(parsed: FieldAttrs) -> syn::Result<FieldAttrs> {
    if let Some(skip) = parsed.skip
        && (parsed.rename.is_some() || parsed.flatten.is_some() || parsed.encoding.is_some())
    {
        return Err(syn::Error::new(
            skip,
            "`skip` cannot be combined with other attributes: a skipped field produces no column",
        ));
    }
    if let (Some((_, rename)), Some(_)) = (&parsed.rename, &parsed.flatten) {
        return Err(syn::Error::new(
            *rename,
            "`rename` cannot be combined with `flatten`: a flattened field adds no name of its \
             own. To put its columns under a name, use `flatten(prefix = \"...\")`",
        ));
    }
    if let (Some((_, flatten)), Some((encoding, _))) = (&parsed.flatten, &parsed.encoding) {
        return Err(syn::Error::new(
            *flatten,
            format!(
                "`flatten` cannot be combined with `{}`: it applies to nested rows, not to leaf values",
                encoding.attribute()
            ),
        ));
    }
    Ok(parsed)
}
