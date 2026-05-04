use crate::ir::{BaseType, DateTimeUnit, FieldIR, PrimitiveTransform, StructIR};
use crate::type_analysis::{
    DEFAULT_DATETIME_UNIT, DEFAULT_DECIMAL_PRECISION, DEFAULT_DECIMAL_SCALE, analyze_type,
};
use quote::format_ident;
use syn::{Data, DeriveInput, Fields, Ident};

/// Mutually-exclusive field-level override declared via `#[df_derive(...)]`.
/// `None` means the field had no override; `derive_transform` injects defaults
/// (e.g. `DateTimeToInt(Milliseconds)` for `chrono::DateTime<Utc>`) in that case.
enum FieldOverride {
    None,
    AsStr,
    AsString,
    Decimal { precision: u8, scale: u8 },
    TimeUnit(DateTimeUnit),
}

fn parse_decimal_attr(meta: &syn::meta::ParseNestedMeta<'_>) -> Result<(u8, u8), syn::Error> {
    let mut precision: Option<u8> = None;
    let mut scale: Option<u8> = None;
    meta.parse_nested_meta(|sub| {
        if sub.path.is_ident("precision") {
            let lit: syn::LitInt = sub.value()?.parse()?;
            let value: u8 = lit.base10_parse().map_err(|_| {
                sub.error("decimal precision must fit in u8 (Polars requires 1..=38)")
            })?;
            if !(1..=38).contains(&value) {
                return Err(sub.error(format!(
                    "decimal precision must satisfy 1 <= precision <= 38 (got {value})"
                )));
            }
            precision = Some(value);
            Ok(())
        } else if sub.path.is_ident("scale") {
            let lit: syn::LitInt = sub.value()?.parse()?;
            let value: u8 = lit
                .base10_parse()
                .map_err(|_| sub.error("decimal scale must fit in u8 (max 38)"))?;
            if value > 38 {
                return Err(sub.error(format!("decimal scale must be <= 38 (got {value})")));
            }
            scale = Some(value);
            Ok(())
        } else {
            Err(sub.error(
                "unknown key inside `decimal(...)`; expected `precision = N` or `scale = N`",
            ))
        }
    })?;
    let p = precision.ok_or_else(|| meta.error("`decimal(...)` requires `precision = N`"))?;
    let s = scale.ok_or_else(|| meta.error("`decimal(...)` requires `scale = N`"))?;
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
        (FieldOverride::AsStr, FieldOverride::AsString)
        | (FieldOverride::AsString, FieldOverride::AsStr) => format!(
            "field `{field_display_name}` has both `as_str` and `as_string`; \
             pick one — `as_str` borrows via `AsRef<str>` (no allocation), \
             `as_string` formats via `Display` (allocates per row)"
        ),
        (FieldOverride::Decimal { .. }, FieldOverride::AsStr | FieldOverride::AsString)
        | (FieldOverride::AsStr | FieldOverride::AsString, FieldOverride::Decimal { .. }) => {
            format!(
                "field `{field_display_name}` combines `decimal(...)` with `as_str`/`as_string`; \
                 `as_str`/`as_string` produce a String column, so the `decimal(...)` \
                 dtype override has no effect — drop one"
            )
        }
        (FieldOverride::TimeUnit(_), FieldOverride::AsStr | FieldOverride::AsString)
        | (FieldOverride::AsStr | FieldOverride::AsString, FieldOverride::TimeUnit(_)) => {
            format!(
                "field `{field_display_name}` combines `time_unit = \"...\"` with \
                 `as_str`/`as_string`; the latter produces a String column, so the \
                 `time_unit` override has no effect — drop one"
            )
        }
        (FieldOverride::Decimal { .. }, FieldOverride::TimeUnit(_))
        | (FieldOverride::TimeUnit(_), FieldOverride::Decimal { .. }) => format!(
            "field `{field_display_name}` combines `decimal(...)` with `time_unit = \"...\"`; \
             pick one — `decimal(...)` only applies to `rust_decimal::Decimal`, \
             `time_unit` only applies to `chrono::DateTime<Utc>`"
        ),
        (FieldOverride::None, _)
        | (_, FieldOverride::None)
        | (FieldOverride::AsStr, FieldOverride::AsStr)
        | (FieldOverride::AsString, FieldOverride::AsString)
        | (FieldOverride::Decimal { .. }, FieldOverride::Decimal { .. })
        | (FieldOverride::TimeUnit(_), FieldOverride::TimeUnit(_)) => unreachable!(
            "override_conflict invoked on a non-conflicting pair; \
             the caller must check existing/incoming variants differ"
        ),
    };
    syn::Error::new_spanned(field, message)
}

/// Set `override_` to `incoming` only if no override has been declared yet;
/// otherwise emit the conflict error for the (existing, incoming) pair. Same-key
/// repeats (`#[df_derive(as_str, as_str)]`, two `decimal(...)` blocks) are
/// idempotent / last-wins respectively, matching pre-refactor de facto behavior.
fn set_override(
    field: &syn::Field,
    field_display_name: &str,
    override_: &mut FieldOverride,
    incoming: FieldOverride,
) -> Result<(), syn::Error> {
    match (&*override_, &incoming) {
        (FieldOverride::None, _)
        | (FieldOverride::AsStr, FieldOverride::AsStr)
        | (FieldOverride::AsString, FieldOverride::AsString)
        | (FieldOverride::Decimal { .. }, FieldOverride::Decimal { .. })
        | (FieldOverride::TimeUnit(_), FieldOverride::TimeUnit(_)) => {
            *override_ = incoming;
            Ok(())
        }
        _ => Err(override_conflict(
            field,
            field_display_name,
            override_,
            &incoming,
        )),
    }
}

fn parse_field_override(
    field: &syn::Field,
    field_display_name: &str,
) -> Result<FieldOverride, syn::Error> {
    let mut override_ = FieldOverride::None;
    for attr in &field.attrs {
        if attr.path().is_ident("df_derive") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("as_string") {
                    set_override(field, field_display_name, &mut override_, FieldOverride::AsString)
                } else if meta.path.is_ident("as_str") {
                    set_override(field, field_display_name, &mut override_, FieldOverride::AsStr)
                } else if meta.path.is_ident("decimal") {
                    let (precision, scale) = parse_decimal_attr(&meta)?;
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::Decimal { precision, scale },
                    )
                } else if meta.path.is_ident("time_unit") {
                    let unit = parse_time_unit_attr(&meta)?;
                    set_override(
                        field,
                        field_display_name,
                        &mut override_,
                        FieldOverride::TimeUnit(unit),
                    )
                } else {
                    Err(meta.error(
                        "unknown key in #[df_derive(...)] field attribute; expected `as_str`, `as_string`, `decimal(precision = N, scale = N)`, or `time_unit = \"ms\"|\"us\"|\"ns\"`",
                    ))
                }
            })?;
        }
    }
    Ok(override_)
}

/// Single source of truth for combining a parsed `FieldOverride` with the
/// analyzed `BaseType` into the final `Option<PrimitiveTransform>` carried on
/// the IR. Performs base-type compatibility checks for every override variant
/// and injects the default `DateTimeToInt(Milliseconds)` /
/// `DecimalToInt128 { 38, 10 }` transforms when no override was declared.
fn derive_transform(
    field: &syn::Field,
    field_display_name: &str,
    override_: &FieldOverride,
    base: &BaseType,
) -> Result<Option<PrimitiveTransform>, syn::Error> {
    match override_ {
        FieldOverride::None => Ok(match base {
            BaseType::DateTimeUtc => Some(PrimitiveTransform::DateTimeToInt(DEFAULT_DATETIME_UNIT)),
            BaseType::Decimal => Some(PrimitiveTransform::DecimalToInt128 {
                precision: DEFAULT_DECIMAL_PRECISION,
                scale: DEFAULT_DECIMAL_SCALE,
            }),
            BaseType::F64
            | BaseType::F32
            | BaseType::I64
            | BaseType::U64
            | BaseType::I32
            | BaseType::U32
            | BaseType::I16
            | BaseType::U16
            | BaseType::I8
            | BaseType::U8
            | BaseType::Bool
            | BaseType::String
            | BaseType::ISize
            | BaseType::USize
            | BaseType::Struct(..)
            | BaseType::Generic(..) => None,
        }),
        FieldOverride::AsString => Ok(Some(PrimitiveTransform::ToString)),
        FieldOverride::AsStr => match base {
            BaseType::String | BaseType::Struct(..) | BaseType::Generic(..) => {
                Ok(Some(PrimitiveTransform::AsStr))
            }
            BaseType::F64
            | BaseType::F32
            | BaseType::I64
            | BaseType::U64
            | BaseType::I32
            | BaseType::U32
            | BaseType::I16
            | BaseType::U16
            | BaseType::I8
            | BaseType::U8
            | BaseType::Bool
            | BaseType::ISize
            | BaseType::USize
            | BaseType::DateTimeUtc
            | BaseType::Decimal => Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{field_display_name}` has `as_str` but its base type does not implement \
                     `AsRef<str>`; `as_str` only applies to `String`, custom struct types, or \
                     generic type parameters — drop the attribute or change the field type"
                ),
            )),
        },
        FieldOverride::Decimal { precision, scale } => match base {
            BaseType::Decimal => Ok(Some(PrimitiveTransform::DecimalToInt128 {
                precision: *precision,
                scale: *scale,
            })),
            BaseType::F64
            | BaseType::F32
            | BaseType::I64
            | BaseType::U64
            | BaseType::I32
            | BaseType::U32
            | BaseType::I16
            | BaseType::U16
            | BaseType::I8
            | BaseType::U8
            | BaseType::Bool
            | BaseType::String
            | BaseType::ISize
            | BaseType::USize
            | BaseType::DateTimeUtc
            | BaseType::Struct(..)
            | BaseType::Generic(..) => Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{field_display_name}` has `decimal(...)` but its base type is not \
                     `rust_decimal::Decimal`; remove the attribute or change the field type"
                ),
            )),
        },
        FieldOverride::TimeUnit(unit) => match base {
            BaseType::DateTimeUtc => Ok(Some(PrimitiveTransform::DateTimeToInt(*unit))),
            BaseType::F64
            | BaseType::F32
            | BaseType::I64
            | BaseType::U64
            | BaseType::I32
            | BaseType::U32
            | BaseType::I16
            | BaseType::U16
            | BaseType::I8
            | BaseType::U8
            | BaseType::Bool
            | BaseType::String
            | BaseType::ISize
            | BaseType::USize
            | BaseType::Decimal
            | BaseType::Struct(..)
            | BaseType::Generic(..) => Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
                     not `chrono::DateTime<Utc>`; remove the attribute or change the field type"
                ),
            )),
        },
    }
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
                let field_name_ident = field
                    .ident
                    .as_ref()
                    .expect("named fields must have ident")
                    .clone();

                let display_name = field_name_ident.to_string();
                let override_ = parse_field_override(field, &display_name)?;
                let analyzed = analyze_type(&field.ty, &generic_params)?;
                let transform = derive_transform(field, &display_name, &override_, &analyzed.base)?;
                fields_ir.push(FieldIR {
                    name: field_name_ident,
                    field_index: None,
                    wrappers: analyzed.wrappers,
                    base_type: analyzed.base,
                    transform,
                    field_ty: field.ty.clone(),
                });
            }
        }
        Fields::Unit => {}
        Fields::Unnamed(unnamed) => {
            for (index, field) in unnamed.unnamed.iter().enumerate() {
                let field_name_ident = format_ident!("field_{}", index);

                let display_name = field_name_ident.to_string();
                let override_ = parse_field_override(field, &display_name)?;
                let analyzed = analyze_type(&field.ty, &generic_params)?;
                let transform = derive_transform(field, &display_name, &override_, &analyzed.base)?;
                fields_ir.push(FieldIR {
                    name: field_name_ident,
                    field_index: Some(index),
                    wrappers: analyzed.wrappers,
                    base_type: analyzed.base,
                    transform,
                    field_ty: field.ty.clone(),
                });
            }
        }
    }

    Ok(StructIR {
        name,
        generics,
        fields: fields_ir,
    })
}
