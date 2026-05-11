use crate::ir::{DateTimeUnit, NumericKind};
use syn::{GenericArgument, Ident, PathArguments, Type, TypePath};

/// Default `Datetime` precision for `chrono::DateTime<Utc>` and
/// `chrono::NaiveDateTime` fields without an explicit `time_unit` override.
/// Matches the historical default this crate shipped with.
pub const DEFAULT_DATETIME_UNIT: DateTimeUnit = DateTimeUnit::Milliseconds;
/// Default `Duration` precision for `std::time::Duration` and
/// `chrono::Duration` (`chrono::TimeDelta`) fields without an explicit
/// `time_unit` override. Nanoseconds is the most-information-preserving
/// choice and matches `polars-arrow`'s default `Duration` representation.
pub const DEFAULT_DURATION_UNIT: DateTimeUnit = DateTimeUnit::Nanoseconds;
/// Default `Decimal(precision, scale)` for fields whose written type path
/// ends in `Decimal` and has no explicit `decimal(...)` override.
pub const DEFAULT_DECIMAL_PRECISION: u8 = 38;
/// Default scale paired with `DEFAULT_DECIMAL_PRECISION`.
pub const DEFAULT_DECIMAL_SCALE: u8 = 10;

/// Raw wrapper position before normalization. The parser collapses these
/// into a `WrapperShape` (with consecutive `Option`s folded per Polars's
/// single-validity-bit-per-level representation).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RawWrapper {
    Option,
    Vec,
}

/// Analyzed base type the parser uses internally before folding through the
/// `(override, base)` legality matrix into a `LeafSpec`. Distinct from the
/// IR's `LeafSpec` because this layer doesn't yet carry override-dependent
/// information (e.g. `Decimal { precision, scale }`, `DateTime(unit)`,
/// stringy classification) — that fusion happens in the parser when the
/// override is consulted.
#[derive(Clone)]
pub enum AnalyzedBase {
    Numeric(NumericKind),
    String,
    Bool,
    DateTimeUtc,
    /// `chrono::NaiveDate` — last-segment `NaiveDate` with no generic args
    /// matches, mirroring `is_datetime_utc`'s leniency. The encoder emits
    /// chrono calls; a same-name false positive surfaces as a compile
    /// error at the call site.
    NaiveDate,
    /// `chrono::NaiveTime` — last-segment `NaiveTime` with no generic args
    /// matches, same leniency as [`Self::NaiveDate`].
    NaiveTime,
    /// `chrono::NaiveDateTime` — last-segment `NaiveDateTime` with no
    /// generic args matches, same leniency as [`Self::NaiveDate`].
    NaiveDateTime,
    /// Exactly `std::time::Duration` or `core::time::Duration`. Detected by
    /// strict path matching to disambiguate from `chrono::Duration` and the
    /// external `time::Duration`; bare `Duration` is rejected as ambiguous in
    /// [`analyze_type`].
    StdDuration,
    /// `chrono::Duration` (alias for `chrono::TimeDelta`). Detected by path
    /// segment matching. Codegen uses the user's declared field-type tokens
    /// directly so type inference resolves the alias correctly.
    ChronoDuration,
    /// Any path whose last segment is `Decimal`. This is deliberately
    /// syntax-based: derive macros cannot resolve type aliases or prove this
    /// is `rust_decimal::Decimal`, so projects such as paft can expose a
    /// backend facade (`paft_decimal::Decimal`) and still get implicit decimal
    /// support. Non-`Decimal` backend names opt in with `decimal(...)`.
    Decimal,
    /// Concrete user-defined struct, with optional angle-bracketed generic
    /// arguments at the field's use site (e.g. `<M>` in `Vec<Foo<M>>`).
    Struct(Ident, Option<syn::AngleBracketedGenericArguments>),
    /// Generic type parameter declared on the enclosing struct.
    Generic(Ident),
    /// Tuple-typed base, with each element recursively analyzed. Empty
    /// tuples (`()`) are rejected at parse time — they contribute zero
    /// columns. Field-level attributes are rejected on every tuple base
    /// because per-element attribute selection isn't expressible at the
    /// field level.
    Tuple(Vec<AnalyzedType>),
}

#[derive(Clone)]
pub struct AnalyzedType {
    pub base: AnalyzedBase,
    pub wrappers: Vec<RawWrapper>,
    /// Original syntactic type token for this analyzed type. The parser
    /// preserves it so per-element trait-bound asserts (`as_str` const-fn
    /// asserts on tuple elements with stringy bases) anchor at the user's
    /// element-type span. For the top-level field, this is the user's field
    /// type; for tuple elements, this is the element's own type.
    pub field_ty: syn::Type,
    /// Number of smart-pointer layers (`Box` / `Rc` / `Arc` / `Cow`) peeled
    /// off the field type ABOVE the first wrapper (`Option` / `Vec`). These
    /// layers are dereffed at the access expression itself — `it.field`
    /// becomes `(*(*(it.field)))` for two outer Boxes — so the rest of the
    /// codegen sees a clean wrapper stack over the inner type.
    pub outer_smart_ptr_depth: usize,
    /// Number of smart-pointer layers peeled BELOW some wrapper but above
    /// the leaf. These cannot be dereffed at the access (you can't deref
    /// `Option<Box<T>>` or `Vec<Arc<T>>`) — instead the encoder injects an
    /// extra deref at the per-element binding inside leaf push bodies.
    /// Method-call autoderef handles most cases (e.g. `arc_string.as_str()`
    /// resolves through Deref), but pattern-binding-by-value sites
    /// (`match #access { Some(v) => buf.push(v); ... }` for numerics) need
    /// the extra `*` to extract the inner value.
    pub inner_smart_ptr_depth: usize,
}

pub fn analyze_type(ty: &Type, generic_params: &[Ident]) -> Result<AnalyzedType, syn::Error> {
    let peeled = peel_type_wrappers(ty)?;

    reject_unsupported_collection_type(peeled.current_type)?;
    reject_bare_duration(peeled.current_type, generic_params)?;

    let base = analyze_base_type(peeled.current_type, generic_params)?;

    Ok(AnalyzedType {
        base,
        wrappers: peeled.wrappers,
        outer_smart_ptr_depth: peeled.outer_smart_ptr_depth,
        inner_smart_ptr_depth: peeled.inner_smart_ptr_depth,
        field_ty: ty.clone(),
    })
}

struct PeeledType<'a> {
    wrappers: Vec<RawWrapper>,
    current_type: &'a Type,
    outer_smart_ptr_depth: usize,
    inner_smart_ptr_depth: usize,
}

const fn bump_smart_ptr_depths(outer: &mut usize, inner: &mut usize, wrappers: &[RawWrapper]) {
    if wrappers.is_empty() {
        *outer += 1;
    } else {
        *inner += 1;
    }
}

fn peel_type_wrappers(ty: &Type) -> Result<PeeledType<'_>, syn::Error> {
    let mut wrappers: Vec<RawWrapper> = Vec::new();
    let mut outer_smart_ptr_depth: usize = 0;
    let mut inner_smart_ptr_depth: usize = 0;
    let mut current_type = ty;

    // Loop to peel off wrappers in any order. Option/Vec push onto the
    // wrapper stack; Box/Rc/Arc/Cow are transparent — they bump the outer
    // depth (codegen rewrites the access expression) before any wrapper is
    // seen, and the inner depth (codegen injects deref at the per-element
    // leaf binding) once a wrapper has been pushed.
    loop {
        if let Some(inner_ty) = extract_inner_type(current_type, "Option") {
            wrappers.push(RawWrapper::Option);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = extract_inner_type(current_type, "Vec") {
            wrappers.push(RawWrapper::Vec);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = extract_inner_type(current_type, "Box")
            .or_else(|| extract_inner_type(current_type, "Rc"))
            .or_else(|| extract_inner_type(current_type, "Arc"))
        {
            bump_smart_ptr_depths(
                &mut outer_smart_ptr_depth,
                &mut inner_smart_ptr_depth,
                &wrappers,
            );
            current_type = inner_ty;
            continue;
        }
        if let Some(action) = peel_cow(current_type) {
            match action {
                CowPeel::Rebind(inner) => {
                    bump_smart_ptr_depths(
                        &mut outer_smart_ptr_depth,
                        &mut inner_smart_ptr_depth,
                        &wrappers,
                    );
                    current_type = inner;
                }
                CowPeel::UnsizedReject => {
                    return Err(syn::Error::new_spanned(
                        current_type,
                        "df-derive does not support `Cow<'_, str>` or `Cow<'_, [T]>` \
                         (unsized inner types). Use the owned type directly: \
                         `String` for borrowed strings, `Vec<T>` for borrowed slices.",
                    ));
                }
            }
            continue;
        }
        // No more wrappers found, break the loop
        break;
    }

    Ok(PeeledType {
        wrappers,
        current_type,
        outer_smart_ptr_depth,
        inner_smart_ptr_depth,
    })
}

fn reject_unsupported_collection_type(current_type: &Type) -> Result<(), syn::Error> {
    // Before resolving the base type, reject a small allow-list of common
    // wrapper / collection types with an actionable hint. These all parse
    // fine as a `Type::Path` and would otherwise either fall through to the
    // generic "Unsupported field type" error or — worse — be silently
    // routed through the `Struct` arm and explode at codegen time.
    if let Type::Path(type_path) = current_type
        && let Some(segment) = type_path.path.segments.last()
    {
        let hint: Option<&'static str> = match segment.ident.to_string().as_str() {
            "HashMap" => Some(
                "df-derive does not support `HashMap` fields. Convert to \
                 `Vec<(K, V)>` or pre-flatten into named columns before assignment.",
            ),
            "BTreeMap" => Some(
                "df-derive does not support `BTreeMap` fields. Convert to \
                 `Vec<(K, V)>` or pre-flatten into named columns before assignment.",
            ),
            "HashSet" => Some(
                "df-derive does not support `HashSet` fields. Convert to \
                 `Vec<T>` (order will be set-defined, not insertion-defined).",
            ),
            _ => None,
        };
        if let Some(message) = hint {
            return Err(syn::Error::new_spanned(current_type, message));
        }
    }
    Ok(())
}

fn reject_bare_duration(current_type: &Type, generic_params: &[Ident]) -> Result<(), syn::Error> {
    // Bare `Duration` (no qualifier, no generic args, not a known generic
    // param) is ambiguous between `std::time::Duration` and
    // `chrono::Duration` — both crates are commonly in scope. Reject with
    // the disambiguation hint anchored at the field's type token.
    if let Type::Path(type_path) = current_type
        && type_path.qself.is_none()
        && type_path.path.segments.len() == 1
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "Duration"
        && matches!(segment.arguments, PathArguments::None)
        && !generic_params.iter().any(|p| p == &segment.ident)
    {
        return Err(syn::Error::new_spanned(
            current_type,
            "bare `Duration` is ambiguous; use `std::time::Duration`, \
             `core::time::Duration`, or `chrono::Duration` to disambiguate",
        ));
    }
    Ok(())
}

fn analyze_base_type(ty: &Type, generic_params: &[Ident]) -> Result<AnalyzedBase, syn::Error> {
    // Tuple bases recurse into element analyses. The empty tuple `()` is
    // rejected here — a unit-typed field contributes zero columns, which
    // collides with the parser's invariant that every field produces at
    // least one schema entry.
    if let Type::Tuple(tup) = ty {
        if tup.elems.is_empty() {
            return Err(syn::Error::new_spanned(
                ty,
                "df-derive does not support unit-typed (`()`) fields; \
                 they contribute zero columns. Remove the field or replace \
                 it with a non-unit type.",
            ));
        }
        let mut elements: Vec<AnalyzedType> = Vec::with_capacity(tup.elems.len());
        for elem in &tup.elems {
            elements.push(analyze_type(elem, generic_params)?);
        }
        return Ok(AnalyzedBase::Tuple(elements));
    }
    if is_datetime_utc(ty) {
        return Ok(AnalyzedBase::DateTimeUtc);
    }
    // Disambiguate `Duration` first (qualified-path matches) — both bases
    // share the last segment `Duration`, so naive last-segment matching is
    // insufficient. Bare `Duration` is rejected upstream in `analyze_type`,
    // not here, so the only `Duration` paths reaching this function are
    // qualified (e.g. `std::time::Duration`, `core::time::Duration`, or
    // `chrono::Duration`).
    if let Type::Path(type_path) = ty {
        if is_std_duration(type_path) {
            return Ok(AnalyzedBase::StdDuration);
        }
        if is_chrono_duration(type_path) {
            return Ok(AnalyzedBase::ChronoDuration);
        }
    }
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        let type_ident = &segment.ident;
        let has_args = !matches!(segment.arguments, PathArguments::None);
        let is_single_segment = type_path.qself.is_none() && type_path.path.segments.len() == 1;
        let base = match type_ident.to_string().as_str() {
            "String" => AnalyzedBase::String,
            "f64" => AnalyzedBase::Numeric(NumericKind::F64),
            "f32" => AnalyzedBase::Numeric(NumericKind::F32),
            "i8" => AnalyzedBase::Numeric(NumericKind::I8),
            "u8" => AnalyzedBase::Numeric(NumericKind::U8),
            "i16" => AnalyzedBase::Numeric(NumericKind::I16),
            "u16" => AnalyzedBase::Numeric(NumericKind::U16),
            "i64" => AnalyzedBase::Numeric(NumericKind::I64),
            "i128" => AnalyzedBase::Numeric(NumericKind::I128),
            "isize" => AnalyzedBase::Numeric(NumericKind::ISize),
            "u64" => AnalyzedBase::Numeric(NumericKind::U64),
            "u128" => AnalyzedBase::Numeric(NumericKind::U128),
            "usize" => AnalyzedBase::Numeric(NumericKind::USize),
            "u32" => AnalyzedBase::Numeric(NumericKind::U32),
            "i32" => AnalyzedBase::Numeric(NumericKind::I32),
            "bool" => AnalyzedBase::Bool,
            "Decimal" => AnalyzedBase::Decimal,
            // Last-segment ident matching, mirroring `is_datetime_utc`'s
            // leniency. `NaiveDate`, `NaiveTime`, and `NaiveDateTime` take
            // no generic arguments, so a re-export under another path still
            // resolves to chrono's type at the call site; if the user's type
            // happens to share the name without sharing the API, the generated
            // chrono method calls fail at compile time at the user's field site.
            "NaiveDate" if !has_args => AnalyzedBase::NaiveDate,
            "NaiveTime" if !has_args => AnalyzedBase::NaiveTime,
            "NaiveDateTime" if !has_args => AnalyzedBase::NaiveDateTime,
            _ => {
                if is_single_segment && !has_args && generic_params.iter().any(|p| p == type_ident)
                {
                    AnalyzedBase::Generic(type_ident.clone())
                } else {
                    let args = match &segment.arguments {
                        PathArguments::AngleBracketed(ab) => Some(ab.clone()),
                        _ => None,
                    };
                    AnalyzedBase::Struct(type_ident.clone(), args)
                }
            }
        };
        return Ok(base);
    }
    Err(syn::Error::new_spanned(ty, "Unsupported field type"))
}

/// Detect exactly `std::time::Duration` or `core::time::Duration`.
///
/// This is intentionally stricter than the `Decimal` heuristic. The Duration
/// encoder emits inherent std/core methods such as `as_nanos()`, so accepting
/// any path ending in `time::Duration` would accidentally capture the external
/// `time` crate's signed duration type.
fn is_std_duration(type_path: &TypePath) -> bool {
    if type_path.qself.is_some() || type_path.path.segments.len() != 3 {
        return false;
    }
    let mut segments = type_path.path.segments.iter();
    let Some(root) = segments.next() else {
        return false;
    };
    let Some(module) = segments.next() else {
        return false;
    };
    let Some(leaf) = segments.next() else {
        return false;
    };
    matches!(root.ident.to_string().as_str(), "std" | "core")
        && root.arguments.is_empty()
        && module.ident == "time"
        && module.arguments.is_empty()
        && leaf.ident == "Duration"
        && leaf.arguments.is_empty()
}

/// Detect `chrono::Duration` or `chrono::TimeDelta`. `chrono::Duration` is
/// a type alias for `chrono::TimeDelta` since chrono 0.4.30; both names
/// resolve to the same impl block, so we accept either tail. Codegen reads
/// the user's declared field-type tokens directly so type inference handles
/// the alias transparently.
fn is_chrono_duration(type_path: &TypePath) -> bool {
    let segs: Vec<String> = type_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    let last = match segs.last() {
        Some(s) if s == "Duration" || s == "TimeDelta" => s,
        _ => return false,
    };
    // For unqualified `Duration`, only accept when the path is rooted at
    // `chrono::` or contains a `chrono` segment somewhere — bare `Duration`
    // is rejected upstream, but a path like `mycrate::Duration` should not
    // route here. `TimeDelta` is chrono-specific enough that the bare-ident
    // case is unlikely to collide; still require an upstream `chrono` for
    // symmetry.
    if last == "Duration" {
        // Accept paths like `chrono::Duration` or `::chrono::Duration`.
        // Reject `std::time::Duration` (handled by `is_std_duration`),
        // `core::time::Duration`, or anything with a `time` segment in
        // the path — those are the std flavor.
        if segs.iter().any(|s| s == "time") {
            return false;
        }
        return segs.iter().any(|s| s == "chrono");
    }
    // `TimeDelta` only lives in chrono.
    segs.iter().any(|s| s == "chrono") || segs.len() == 1
}

fn extract_inner_type<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == wrapper
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
    {
        return Some(inner_ty);
    }
    None
}

/// Outcome of peeling a `Cow<'a, T>` layer. Cow's first generic argument is
/// the lifetime, not the inner type, so the standard `extract_inner_type`
/// helper (which keys on `args.first()`) cannot be used. The two unsized
/// inner forms (`str` and `[T]`) are rejected at parse time with an
/// actionable hint pointing at the owned form (`String` / `Vec<T>`); the
/// sized inner case rebinds and continues the peel loop.
enum CowPeel<'a> {
    /// `Cow<'a, OwnedT>` where `OwnedT: Sized` — rebind to `OwnedT`.
    Rebind(&'a Type),
    /// `Cow<'a, str>` or `Cow<'a, [T]>` — the inner is unsized. The
    /// existing codegen invariants (every leaf method-call path resolves
    /// through `AsRef<str>` or `Deref` autoderef) collapse around `str`'s
    /// unstable inherent `as_str()` and `[T]`'s lack of `Vec`-shaped
    /// methods, so the parser rejects these forms at parse time. Use the
    /// owned type directly (`String` or `Vec<T>`).
    UnsizedReject,
}

/// Peel a `Cow<'a, T>` layer if the type is one. Last-segment ident match
/// (mirrors `extract_inner_type`'s leniency for qualified paths like
/// `std::borrow::Cow`). Returns `None` for non-Cow types.
fn peel_cow(ty: &Type) -> Option<CowPeel<'_>> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Cow" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    // Cow takes `<'a, T>` — find the first `Type` argument (skipping
    // the leading lifetime). Defensive against generic-syntax variation:
    // any GenericArgument::Type wins regardless of position.
    let inner_ty = args.args.iter().find_map(|a| match a {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    })?;
    // `Cow<str>` or `Cow<[T]>` — unsized inner, reject at parse time.
    // The autoderef chain for `Cow<str>::as_str()` ends at `str`'s
    // unstable inherent method, and `Cow<[T]>` needs Vec-shaped iteration
    // we don't synthesize. Users should use `String` / `Vec<T>` directly.
    if let Type::Path(inner_path) = inner_ty
        && inner_path.qself.is_none()
        && inner_path.path.segments.len() == 1
        && let Some(seg) = inner_path.path.segments.last()
        && seg.ident == "str"
        && matches!(seg.arguments, PathArguments::None)
    {
        return Some(CowPeel::UnsizedReject);
    }
    if matches!(inner_ty, Type::Slice(_)) {
        return Some(CowPeel::UnsizedReject);
    }
    Some(CowPeel::Rebind(inner_ty))
}

/// Detect a `chrono::DateTime<Utc>` field by ident only.
///
/// The match looks at the last segment of the outer path (`DateTime`) and the
/// last segment of the first generic argument's path (`Utc`). Anything that
/// happens to share those idents — e.g. `some_other_crate::DateTime<other::Utc>`
/// — would be a false positive and routed through the chrono encoder.
///
/// This leniency is intentional: user crates frequently re-export
/// `chrono::DateTime<chrono::Utc>` under their own paths (type aliases, prelude
/// modules, glob re-exports), and tightening the match to a specific path
/// prefix would break those uses without a robust way to recover the original
/// definition from a `syn::Type` alone. The proc-macro happily generates code
/// that calls chrono's `timestamp_*` methods; if the type isn't actually
/// chrono's, the user gets a compile error at the call site, which is the
/// correct failure mode.
fn is_datetime_utc(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "DateTime"
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(Type::Path(inner))) = args.args.first()
        && let Some(inner_seg) = inner.path.segments.last()
    {
        return inner_seg.ident == "Utc";
    }
    false
}
