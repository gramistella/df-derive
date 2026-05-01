use crate::ir::{BaseType, DateTimeUnit, FieldIR, PrimitiveTransform, StructIR};
use crate::type_analysis::analyze_type;
use quote::format_ident;
use syn::{Data, DeriveInput, Fields, Ident};

#[derive(Default, Clone, Copy)]
struct FieldAttributes {
    as_string: bool,
    as_str: bool,
    decimal: Option<(u8, u8)>,
    time_unit: Option<DateTimeUnit>,
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

fn parse_attributes(
    field: &syn::Field,
    field_display_name: &str,
) -> Result<FieldAttributes, syn::Error> {
    let mut attrs = FieldAttributes::default();
    for attr in &field.attrs {
        if attr.path().is_ident("df_derive") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("as_string") {
                    attrs.as_string = true;
                    Ok(())
                } else if meta.path.is_ident("as_str") {
                    attrs.as_str = true;
                    Ok(())
                } else if meta.path.is_ident("decimal") {
                    let (p, s) = parse_decimal_attr(&meta)?;
                    attrs.decimal = Some((p, s));
                    Ok(())
                } else if meta.path.is_ident("time_unit") {
                    attrs.time_unit = Some(parse_time_unit_attr(&meta)?);
                    Ok(())
                } else {
                    // Reject unknown keys so a typo (e.g. `as_strg` or `as_string` swapped
                    // with `as_str`) surfaces at the user's source instead of being
                    // silently ignored.
                    Err(meta.error(
                        "unknown key in #[df_derive(...)] field attribute; expected `as_str`, `as_string`, `decimal(precision = N, scale = N)`, or `time_unit = \"ms\"|\"us\"|\"ns\"`",
                    ))
                }
            })?;
        }
    }
    if attrs.as_string && attrs.as_str {
        return Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has both `as_str` and `as_string`; \
                 pick one — `as_str` borrows via `AsRef<str>` (no allocation), \
                 `as_string` formats via `Display` (allocates per row)"
            ),
        ));
    }
    Ok(attrs)
}

/// Reject `#[df_derive(as_str)]` on a base type that codegen knows cannot be
/// `AsRef<str>`. Without this check, the silent fallback in
/// `generate_inner_series_from_vec` would emit UFCS-against-`String` tokens
/// and lean on the per-field `AsRef<str>` const-fn assert in `helpers.rs` to
/// fail compilation. Surfacing the mistake at parse time yields a cleaner
/// span and message at the user's attribute, not deep in macro expansion.
/// `Struct` / `Generic` / `String` bases are still allowed: the first two are
/// validated by the runtime assert (the parser cannot know whether a user
/// type implements `AsRef<str>`), and `String` itself implements it.
fn reject_as_str_on_incompatible_base(
    field: &syn::Field,
    field_display_name: &str,
    attrs: FieldAttributes,
    base: &BaseType,
) -> Result<(), syn::Error> {
    if !attrs.as_str {
        return Ok(());
    }
    match base {
        BaseType::String | BaseType::Struct(..) | BaseType::Generic(..) => Ok(()),
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
    }
}

/// Apply user-specified `decimal(...)` / `time_unit = "..."` overrides to the
/// transform produced by `analyze_type`. Rejects combinations that don't make
/// sense (e.g. `decimal(...)` on a non-`Decimal` field, or alongside
/// `as_str`/`as_string`).
fn apply_dtype_overrides(
    field: &syn::Field,
    field_display_name: &str,
    attrs: FieldAttributes,
    base: &BaseType,
    transform: &mut Option<PrimitiveTransform>,
) -> Result<(), syn::Error> {
    if let Some((precision, scale)) = attrs.decimal {
        if attrs.as_str || attrs.as_string {
            return Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{field_display_name}` combines `decimal(...)` with `as_str`/`as_string`; \
                     `as_str`/`as_string` produce a String column, so the `decimal(...)` \
                     dtype override has no effect — drop one"
                ),
            ));
        }
        if !matches!(base, BaseType::Decimal) {
            return Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{field_display_name}` has `decimal(...)` but its base type is not \
                     `rust_decimal::Decimal`; remove the attribute or change the field type"
                ),
            ));
        }
        *transform = Some(PrimitiveTransform::DecimalToString { precision, scale });
    }
    if let Some(unit) = attrs.time_unit {
        if attrs.as_str || attrs.as_string {
            return Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{field_display_name}` combines `time_unit = \"...\"` with \
                     `as_str`/`as_string`; the latter produces a String column, so the \
                     `time_unit` override has no effect — drop one"
                ),
            ));
        }
        if !matches!(base, BaseType::DateTimeUtc) {
            return Err(syn::Error::new_spanned(
                field,
                format!(
                    "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
                     not `chrono::DateTime<Utc>`; remove the attribute or change the field type"
                ),
            ));
        }
        *transform = Some(PrimitiveTransform::DateTimeToInt(unit));
    }
    Ok(())
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
                let attrs = parse_attributes(field, &display_name)?;
                let analyzed =
                    analyze_type(&field.ty, attrs.as_string, attrs.as_str, &generic_params)?;

                let base_type = analyzed.base.clone();
                let mut transform = analyzed.transform.clone();
                reject_as_str_on_incompatible_base(field, &display_name, attrs, &base_type)?;
                apply_dtype_overrides(field, &display_name, attrs, &base_type, &mut transform)?;
                fields_ir.push(FieldIR {
                    name: field_name_ident,
                    field_index: None,
                    wrappers: analyzed.wrappers.clone(),
                    base_type,
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
                let attrs = parse_attributes(field, &display_name)?;
                let analyzed =
                    analyze_type(&field.ty, attrs.as_string, attrs.as_str, &generic_params)?;

                let base_type = analyzed.base.clone();
                let mut transform = analyzed.transform.clone();
                reject_as_str_on_incompatible_base(field, &display_name, attrs, &base_type)?;
                apply_dtype_overrides(field, &display_name, attrs, &base_type, &mut transform)?;
                fields_ir.push(FieldIR {
                    name: field_name_ident,
                    field_index: Some(index),
                    wrappers: analyzed.wrappers.clone(),
                    base_type,
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
