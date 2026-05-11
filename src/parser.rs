use crate::ir::{
    DateTimeUnit, DurationSource, FieldIR, LeafSpec, NumericKind, StringyBase, StructIR,
    TupleElement, VecLayerSpec, VecLayers, WrapperShape,
};
use crate::type_analysis::{
    AnalyzedBase, DEFAULT_DATETIME_UNIT, DEFAULT_DECIMAL_PRECISION, DEFAULT_DECIMAL_SCALE,
    DEFAULT_DURATION_UNIT, RawWrapper, analyze_type,
};
use quote::format_ident;
use syn::{Data, DeriveInput, Fields, Ident};

/// Mutually-exclusive field-level override declared via `#[df_derive(...)]`.
/// `None` means the field had no override; `parse_leaf_spec` injects defaults
/// (e.g. `DateTimeToInt(Milliseconds)` for `chrono::DateTime<Utc>`) in that case.
enum FieldOverride {
    None,
    AsStr,
    AsString,
    AsBinary,
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
             `time_unit` only applies to `chrono::DateTime<Utc>`, \
             `std::time::Duration`, or `chrono::Duration`"
        ),
        (FieldOverride::AsBinary, _) | (_, FieldOverride::AsBinary) => format!(
            "field `{field_display_name}` combines `as_binary` with another override; \
             `as_binary` produces a Binary column over a `Vec<u8>` shape and is \
             mutually exclusive with `as_str`, `as_string`, `decimal(...)`, and \
             `time_unit = \"...\"` — drop one"
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
        | (FieldOverride::AsBinary, FieldOverride::AsBinary)
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
                } else if meta.path.is_ident("as_binary") {
                    set_override(field, field_display_name, &mut override_, FieldOverride::AsBinary)
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
                        "unknown key in #[df_derive(...)] field attribute; expected `as_str`, `as_string`, `as_binary`, `decimal(precision = N, scale = N)`, or `time_unit = \"ms\"|\"us\"|\"ns\"`",
                    ))
                }
            })?;
        }
    }
    Ok(override_)
}

/// Single source of truth for combining a parsed `FieldOverride` with the
/// analyzed base type into the final `LeafSpec` carried on the IR. Performs
/// base-type compatibility checks for every override variant and injects
/// the default semantics (`DateTimeToInt(Milliseconds)` for
/// `chrono::DateTime<Utc>`, `Decimal(38, 10)` for `rust_decimal::Decimal`)
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
    override_: &FieldOverride,
    base: AnalyzedBase,
) -> Result<LeafSpec, syn::Error> {
    if let AnalyzedBase::Tuple(_) = &base {
        reject_attrs_on_tuple(field, field_display_name, override_)?;
    }
    match override_ {
        FieldOverride::None => Ok(default_leaf_for_base(base)),
        FieldOverride::AsString => Ok(LeafSpec::AsString),
        FieldOverride::AsStr => parse_leaf_as_str(field, field_display_name, base),
        FieldOverride::AsBinary => {
            unreachable!("AsBinary handled by process_field before parse_leaf_spec runs")
        }
        FieldOverride::Decimal { precision, scale } => {
            parse_leaf_decimal(field, field_display_name, &base, *precision, *scale)
        }
        FieldOverride::TimeUnit(unit) => {
            parse_leaf_time_unit(field, field_display_name, &base, *unit)
        }
    }
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
    override_: &FieldOverride,
) -> Result<(), syn::Error> {
    let attr = match override_ {
        FieldOverride::None => return Ok(()),
        FieldOverride::AsStr => "as_str",
        FieldOverride::AsString => "as_string",
        FieldOverride::AsBinary => "as_binary",
        FieldOverride::Decimal { .. } => "decimal(...)",
        FieldOverride::TimeUnit(_) => "time_unit = \"...\"",
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
/// for `DateTime<Utc>`, `Nanoseconds` for `Duration`, `Decimal(38, 10)`,
/// etc. Tuple bases recurse: each element runs through the same default
/// pipeline (no field-level overrides apply at element level — the
/// parser rejects them on the parent field).
fn default_leaf_for_base(base: AnalyzedBase) -> LeafSpec {
    match base {
        AnalyzedBase::Numeric(kind) => LeafSpec::Numeric(kind),
        AnalyzedBase::String => LeafSpec::String,
        AnalyzedBase::Bool => LeafSpec::Bool,
        AnalyzedBase::DateTimeUtc => LeafSpec::DateTime(DEFAULT_DATETIME_UNIT),
        AnalyzedBase::NaiveDate => LeafSpec::NaiveDate,
        AnalyzedBase::NaiveTime => LeafSpec::NaiveTime,
        AnalyzedBase::StdDuration => LeafSpec::Duration {
            unit: DEFAULT_DURATION_UNIT,
            source: DurationSource::Std,
        },
        AnalyzedBase::ChronoDuration => LeafSpec::Duration {
            unit: DEFAULT_DURATION_UNIT,
            source: DurationSource::Chrono,
        },
        AnalyzedBase::Decimal => LeafSpec::Decimal {
            precision: DEFAULT_DECIMAL_PRECISION,
            scale: DEFAULT_DECIMAL_SCALE,
        },
        AnalyzedBase::Struct(ident, args) => LeafSpec::Struct(ident, args),
        AnalyzedBase::Generic(ident) => LeafSpec::Generic(ident),
        AnalyzedBase::Tuple(elements) => {
            let lowered = elements
                .into_iter()
                .map(analyzed_to_tuple_element)
                .collect();
            LeafSpec::Tuple(lowered)
        }
    }
}

/// Lower one analyzed tuple element to its IR form. Recurses into nested
/// tuples (`((i32, String), bool)`), preserves smart-pointer counts on the
/// element, and normalizes the element's wrapper stack independently of the
/// parent's. Field-level attributes are not applied here — they are rejected
/// on the parent field by [`reject_attrs_on_tuple`] before this runs.
fn analyzed_to_tuple_element(analyzed: crate::type_analysis::AnalyzedType) -> TupleElement {
    let leaf_spec = default_leaf_for_base(analyzed.base);
    let wrapper_shape = normalize_wrappers(&analyzed.wrappers);
    TupleElement {
        leaf_spec,
        wrapper_shape,
        field_ty: analyzed.field_ty,
        outer_smart_ptr_depth: analyzed.outer_smart_ptr_depth,
        inner_smart_ptr_depth: analyzed.inner_smart_ptr_depth,
    }
}

fn parse_leaf_as_str(
    field: &syn::Field,
    field_display_name: &str,
    base: AnalyzedBase,
) -> Result<LeafSpec, syn::Error> {
    match base {
        AnalyzedBase::String => Ok(LeafSpec::AsStr(StringyBase::String)),
        AnalyzedBase::Struct(ident, args) => Ok(LeafSpec::AsStr(StringyBase::Struct(ident, args))),
        AnalyzedBase::Generic(ident) => Ok(LeafSpec::AsStr(StringyBase::Generic(ident))),
        // Tuple bases reach this dispatcher only when `parse_leaf_spec`'s
        // upstream `reject_attrs_on_tuple` was bypassed, which it isn't —
        // but the match must be exhaustive on `AnalyzedBase`. Surface a
        // distinct error if it ever does fire.
        AnalyzedBase::Tuple(_)
        | AnalyzedBase::Numeric(_)
        | AnalyzedBase::Bool
        | AnalyzedBase::DateTimeUtc
        | AnalyzedBase::NaiveDate
        | AnalyzedBase::NaiveTime
        | AnalyzedBase::StdDuration
        | AnalyzedBase::ChronoDuration
        | AnalyzedBase::Decimal => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `as_str` but its base type does not implement \
                 `AsRef<str>`; `as_str` only applies to `String`, custom struct types, or \
                 generic type parameters — drop the attribute or change the field type"
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
        AnalyzedBase::Decimal => Ok(LeafSpec::Decimal { precision, scale }),
        _ => Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has `decimal(...)` but its base type is not \
                 `rust_decimal::Decimal`; remove the attribute or change the field type"
            ),
        )),
    }
}

fn parse_leaf_time_unit(
    field: &syn::Field,
    field_display_name: &str,
    base: &AnalyzedBase,
    unit: DateTimeUnit,
) -> Result<LeafSpec, syn::Error> {
    match base {
        AnalyzedBase::DateTimeUtc => Ok(LeafSpec::DateTime(unit)),
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
                 not `chrono::DateTime<Utc>`, `std::time::Duration`, or `chrono::Duration`; \
                 remove the attribute or change the field type"
            ),
        )),
    }
}

/// Normalize the raw outer-to-inner `RawWrapper` sequence into a
/// `WrapperShape` the encoder consumes directly. Consecutive `Option`s
/// collapse into per-position counts: above each `Vec`, immediately
/// surrounding the leaf, or for the leaf-only path, all in one bucket.
/// Polars folds them into a single validity bit per position, so the count
/// is preserved only so the encoder can pick between a direct match
/// (`option_layers == 1`) and the multi-Option `as_ref().and_then(...)`
/// collapse (`option_layers >= 2`).
fn normalize_wrappers(wrappers: &[RawWrapper]) -> WrapperShape {
    let mut layers: Vec<VecLayerSpec> = Vec::new();
    let mut pending_options: usize = 0;
    let mut inner_option_layers: usize = 0;
    let mut saw_vec = false;
    for w in wrappers {
        match w {
            RawWrapper::Option => {
                if saw_vec {
                    inner_option_layers += 1;
                } else {
                    pending_options += 1;
                }
            }
            RawWrapper::Vec => {
                saw_vec = true;
                // Options accumulated since the last Vec wrap THIS Vec from
                // the previous Vec's element perspective: from the new Vec's
                // POV they sit immediately above it as list-level validity.
                // Drop them into the new layer instead of discarding.
                layers.push(VecLayerSpec {
                    option_layers_above: pending_options + std::mem::take(&mut inner_option_layers),
                });
                pending_options = 0;
            }
        }
    }
    if layers.is_empty() {
        return WrapperShape::Leaf {
            option_layers: pending_options,
        };
    }
    WrapperShape::Vec(VecLayers {
        layers,
        inner_option_layers,
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
    generic_params: &[Ident],
) -> Result<FieldIR, syn::Error> {
    let display_name = name_ident.to_string();
    let override_ = parse_field_override(field, &display_name)?;
    let analyzed = analyze_type(&field.ty, generic_params)?;
    let outer_smart_ptr_depth = analyzed.outer_smart_ptr_depth;
    let inner_smart_ptr_depth = analyzed.inner_smart_ptr_depth;
    let (leaf_spec, wrapper_shape) = if matches!(override_, FieldOverride::AsBinary) {
        // `as_binary` over a tuple is rejected here too — `parse_as_binary_shape`
        // only checks the leaf base, but the tuple itself fails the same
        // multi-column attribute rule as every other field-level attribute.
        if matches!(analyzed.base, AnalyzedBase::Tuple(_)) {
            reject_attrs_on_tuple(field, &display_name, &override_)?;
        }
        let (leaf, trimmed) =
            parse_as_binary_shape(field, &display_name, &analyzed.base, &analyzed.wrappers)?;
        (leaf, normalize_wrappers(&trimmed))
    } else {
        let leaf = parse_leaf_spec(field, &display_name, &override_, analyzed.base)?;
        (leaf, normalize_wrappers(&analyzed.wrappers))
    };
    Ok(FieldIR {
        name: name_ident,
        field_index,
        leaf_spec,
        wrapper_shape,
        field_ty: field.ty.clone(),
        outer_smart_ptr_depth,
        inner_smart_ptr_depth,
    })
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
                fields_ir.push(process_field(field, name_ident, None, &generic_params)?);
            }
        }
        Fields::Unit => {}
        Fields::Unnamed(unnamed) => {
            for (index, field) in unnamed.unnamed.iter().enumerate() {
                let name_ident = format_ident!("field_{}", index);
                fields_ir.push(process_field(
                    field,
                    name_ident,
                    Some(index),
                    &generic_params,
                )?);
            }
        }
    }

    Ok(StructIR {
        name,
        generics,
        fields: fields_ir,
    })
}
