use crate::ir::{DateTimeUnit, NumericKind};
use syn::{GenericArgument, Ident, PathArguments, Type, TypePath};

/// Default `Datetime` precision for `chrono::DateTime<Tz>` and
/// `chrono::NaiveDateTime` fields without an explicit `time_unit` override.
/// Matches the historical default this crate shipped with.
pub const DEFAULT_DATETIME_UNIT: DateTimeUnit = DateTimeUnit::Milliseconds;
/// Default `Duration` precision for `std::time::Duration` and
/// `chrono::Duration` (`chrono::TimeDelta`) fields without an explicit
/// `time_unit` override. Nanoseconds is the most-information-preserving
/// choice and matches `polars-arrow`'s default `Duration` representation.
pub const DEFAULT_DURATION_UNIT: DateTimeUnit = DateTimeUnit::Nanoseconds;
/// Default `Decimal(precision, scale)` for bare `Decimal` or
/// `rust_decimal::Decimal` fields without an explicit `decimal(...)` override.
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
    SmartPtr,
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
    /// `&str` — kept as a semantic string leaf instead of peeling to
    /// unsized `str`, mirroring `Cow<'_, str>`.
    BorrowedStr,
    /// `Cow<'_, str>` — kept as a semantic string leaf instead of peeling to
    /// unsized `str` so codegen can borrow via `Cow::as_ref`.
    CowStr,
    /// `&[u8]` — supported only with `#[df_derive(as_binary)]`.
    BorrowedBytes,
    /// `Cow<'_, [u8]>` — supported only with `#[df_derive(as_binary)]`.
    CowBytes,
    /// `&[T]` for non-`u8` element types. Kept as a semantic base so
    /// the parser can emit a domain-specific error instead of returning a
    /// generic unsupported-type diagnostic.
    BorrowedSlice,
    /// `Cow<'_, [T]>` for non-`u8` element types. Kept as a semantic base so
    /// the parser can emit a domain-specific error instead of routing it as a
    /// struct or generic fallback.
    CowSlice,
    Bool,
    /// `chrono::DateTime<Tz>` — syntax-detected for bare `DateTime<Tz>` or
    /// canonical `chrono::DateTime<Tz>` paths. The encoder stores the UTC
    /// instant via chrono's `timestamp_*` methods and materializes
    /// `Datetime(unit, None)`.
    DateTimeTz,
    /// `chrono::NaiveDate` — bare `NaiveDate` or canonical
    /// `chrono::NaiveDate`.
    NaiveDate,
    /// `chrono::NaiveTime` — bare `NaiveTime` or canonical
    /// `chrono::NaiveTime`.
    NaiveTime,
    /// `chrono::NaiveDateTime` — bare `NaiveDateTime` or canonical
    /// `chrono::NaiveDateTime`.
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
    /// Bare `Decimal` or canonical `rust_decimal::Decimal`. Other decimal
    /// backend names opt in with `decimal(...)`.
    Decimal,
    /// Concrete user-defined struct type as written at the field's use site
    /// (for example `Foo`, `models::Foo`, `models::Foo<M>`, or
    /// `<T as Trait>::Item`).
    Struct(Type),
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
    /// Number of transparent pointer layers (`Box` / `Rc` / `Arc` / `Cow` /
    /// borrowed references) peeled
    /// off the field type ABOVE the first wrapper (`Option` / `Vec`). These
    /// layers are dereffed at the access expression itself — `it.field`
    /// becomes `(*(*(it.field)))` for two outer Boxes — so the rest of the
    /// codegen sees a clean wrapper stack over the inner type.
    pub outer_smart_ptr_depth: usize,
}

fn bare_generic_param_ident(ty: &Type, generic_params: &[Ident]) -> Option<Ident> {
    let Type::Path(type_path) = ty else {
        return None;
    };

    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }

    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }

    generic_params
        .iter()
        .any(|param| param == &segment.ident)
        .then(|| segment.ident.clone())
}

pub fn analyze_type(ty: &Type, generic_params: &[Ident]) -> Result<AnalyzedType, syn::Error> {
    let peeled = peel_type_wrappers(ty);

    if bare_generic_param_ident(peeled.current_type, generic_params).is_none() {
        reject_unsupported_collection_type(peeled.current_type)?;
        reject_bare_duration(peeled.current_type, generic_params)?;
    }

    let base = analyze_base_type(peeled.current_type, generic_params)?;

    Ok(AnalyzedType {
        base,
        wrappers: peeled.wrappers,
        outer_smart_ptr_depth: peeled.outer_smart_ptr_depth,
        field_ty: ty.clone(),
    })
}

struct PeeledType<'a> {
    wrappers: Vec<RawWrapper>,
    current_type: &'a Type,
    outer_smart_ptr_depth: usize,
}

fn record_smart_ptr_layer(outer: &mut usize, wrappers: &mut Vec<RawWrapper>) {
    if wrappers.is_empty() {
        *outer += 1;
    } else {
        wrappers.push(RawWrapper::SmartPtr);
    }
}

fn peel_type_wrappers(ty: &Type) -> PeeledType<'_> {
    let mut wrappers: Vec<RawWrapper> = Vec::new();
    let mut outer_smart_ptr_depth: usize = 0;
    let mut current_type = ty;

    // Loop to peel off wrappers in any order. Option/Vec push onto the
    // wrapper stack; Box/Rc/Arc/Cow and borrowed references are transparent
    // when their inner type is sized — they bump the outer depth (codegen
    // rewrites the access expression) before any wrapper is seen, or become
    // `RawWrapper::SmartPtr` once a semantic wrapper has been pushed so the
    // parser can retain the exact boundary where the smart pointer occurred.
    // Borrowed `str` and slices stay as semantic bases because they are
    // unsized and need domain-specific parser decisions.
    loop {
        if let Some(inner_ty) =
            extract_inner_type(current_type, "Option", &["std", "option", "Option"]).or_else(|| {
                extract_inner_type(current_type, "Option", &["core", "option", "Option"])
            })
        {
            wrappers.push(RawWrapper::Option);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = extract_inner_type(current_type, "Vec", &["std", "vec", "Vec"])
            .or_else(|| extract_inner_type(current_type, "Vec", &["alloc", "vec", "Vec"]))
        {
            wrappers.push(RawWrapper::Vec);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = extract_inner_type(current_type, "Box", &["std", "boxed", "Box"])
            .or_else(|| extract_inner_type(current_type, "Box", &["alloc", "boxed", "Box"]))
            .or_else(|| extract_inner_type(current_type, "Rc", &["std", "rc", "Rc"]))
            .or_else(|| extract_inner_type(current_type, "Rc", &["alloc", "rc", "Rc"]))
            .or_else(|| extract_inner_type(current_type, "Arc", &["std", "sync", "Arc"]))
            .or_else(|| extract_inner_type(current_type, "Arc", &["alloc", "sync", "Arc"]))
        {
            record_smart_ptr_layer(&mut outer_smart_ptr_depth, &mut wrappers);
            current_type = inner_ty;
            continue;
        }
        if let Some(action) = peel_cow(current_type) {
            match action {
                CowPeel::Rebind(inner) => {
                    record_smart_ptr_layer(&mut outer_smart_ptr_depth, &mut wrappers);
                    current_type = inner;
                    continue;
                }
                CowPeel::KeepAsSemanticBase => break,
            }
        }
        if let Type::Reference(reference) = current_type {
            if borrowed_reference_base(reference).is_some() {
                break;
            }
            record_smart_ptr_layer(&mut outer_smart_ptr_depth, &mut wrappers);
            current_type = reference.elem.as_ref();
            continue;
        }
        // No more wrappers found, break the loop
        break;
    }

    PeeledType {
        wrappers,
        current_type,
        outer_smart_ptr_depth,
    }
}

fn reject_unsupported_collection_type(current_type: &Type) -> Result<(), syn::Error> {
    // Before resolving the base type, reject a small allow-list of common
    // wrapper / collection types with an actionable hint. These all parse
    // fine as a `Type::Path` and would otherwise either fall through to the
    // generic "Unsupported field type" error or — worse — be silently
    // routed through the `Struct` arm and explode at codegen time.
    if let Type::Path(type_path) = current_type
        && let Some(collection) = unsupported_collection_name(type_path)
    {
        let hint: Option<&'static str> = match collection {
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
    // Tuple bases recurse into element analyses. A direct unit-typed field
    // `field: ()` is rejected here because it would contribute zero columns,
    // colliding with the parser's invariant that every syntactic field
    // produces at least one schema entry. Generic payloads that instantiate to
    // `()` are still supported through the runtime's `ToDataFrame for ()`.
    if let Type::Tuple(tup) = ty {
        if tup.elems.is_empty() {
            return Err(syn::Error::new_spanned(
                ty,
                "df-derive does not support direct unit-typed (`()`) fields; \
                 they would contribute zero columns. Remove the field, replace \
                 it with a non-unit type, or use a generic payload such as \
                 `field: M` with `M = ()`.",
            ));
        }
        let mut elements: Vec<AnalyzedType> = Vec::with_capacity(tup.elems.len());
        for elem in &tup.elems {
            elements.push(analyze_type(elem, generic_params)?);
        }
        return Ok(AnalyzedBase::Tuple(elements));
    }
    if let Type::Reference(reference) = ty
        && let Some(base) = borrowed_reference_base(reference)
    {
        return Ok(base);
    }
    if let Some(ident) = bare_generic_param_ident(ty, generic_params) {
        return Ok(AnalyzedBase::Generic(ident));
    }
    // Disambiguate `Duration` first (qualified-path matches) — both bases
    // share the last segment `Duration`, so naive last-segment matching is
    // insufficient. Bare `Duration` is rejected upstream in `analyze_type`,
    // not here, so the only `Duration` paths reaching this function are
    // qualified (e.g. `std::time::Duration`, `core::time::Duration`, or
    // `chrono::Duration`).
    if let Type::Path(type_path) = ty {
        if let Some(base) = analyze_cow_base(type_path) {
            return Ok(base);
        }
        if is_std_duration(type_path) {
            return Ok(AnalyzedBase::StdDuration);
        }
        if is_chrono_duration(type_path) {
            return Ok(AnalyzedBase::ChronoDuration);
        }
    }
    if let Type::Path(type_path) = ty {
        if is_chrono_datetime(type_path) {
            return Ok(AnalyzedBase::DateTimeTz);
        }
        if let Some(kind) = bare_numeric_kind(type_path).or_else(|| nonzero_numeric_kind(type_path))
        {
            return Ok(AnalyzedBase::Numeric(kind));
        }
        if path_is_exact_no_args(type_path, &["bool"]) {
            return Ok(AnalyzedBase::Bool);
        }
        if is_string_type(type_path) {
            return Ok(AnalyzedBase::String);
        }
        if is_decimal_type(type_path) {
            return Ok(AnalyzedBase::Decimal);
        }
        if is_chrono_no_args_type(type_path, "NaiveDate") {
            return Ok(AnalyzedBase::NaiveDate);
        }
        if is_chrono_no_args_type(type_path, "NaiveTime") {
            return Ok(AnalyzedBase::NaiveTime);
        }
        if is_chrono_no_args_type(type_path, "NaiveDateTime") {
            return Ok(AnalyzedBase::NaiveDateTime);
        }
        return Ok(AnalyzedBase::Struct(ty.clone()));
    }
    Err(syn::Error::new_spanned(ty, "Unsupported field type"))
}

fn path_is_exact_no_args(type_path: &TypePath, segments: &[&str]) -> bool {
    type_path.qself.is_none()
        && type_path.path.segments.len() == segments.len()
        && type_path
            .path
            .segments
            .iter()
            .zip(segments)
            .all(|(segment, expected)| {
                segment.ident == *expected && matches!(segment.arguments, PathArguments::None)
            })
}

fn path_is_exact_with_leaf_args(type_path: &TypePath, segments: &[&str]) -> bool {
    type_path.qself.is_none()
        && type_path.path.segments.len() == segments.len()
        && type_path
            .path
            .segments
            .iter()
            .zip(segments)
            .enumerate()
            .all(|(idx, (segment, expected))| {
                segment.ident == *expected
                    && (idx + 1 == segments.len()
                        || matches!(segment.arguments, PathArguments::None))
            })
}

fn unsupported_collection_name(type_path: &TypePath) -> Option<&'static str> {
    ["HashMap", "BTreeMap", "HashSet"]
        .into_iter()
        .find(|&name| path_is_bare_or_std_collection(type_path, name))
}

fn path_is_bare_or_std_collection(type_path: &TypePath, leaf: &str) -> bool {
    type_path.qself.is_none()
        && ((type_path.path.segments.len() == 1
            && type_path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == leaf))
            || (type_path.path.segments.len() == 3
                && path_prefix_is_no_args(type_path, &["std", "collections"])
                && type_path
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == leaf)))
}

fn path_prefix_is_no_args(type_path: &TypePath, prefix: &[&str]) -> bool {
    type_path.path.segments.len() > prefix.len()
        && type_path
            .path
            .segments
            .iter()
            .zip(prefix)
            .all(|(segment, expected)| {
                segment.ident == *expected && matches!(segment.arguments, PathArguments::None)
            })
}

fn bare_numeric_kind(type_path: &TypePath) -> Option<NumericKind> {
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    match segment.ident.to_string().as_str() {
        "f64" => Some(NumericKind::F64),
        "f32" => Some(NumericKind::F32),
        "i8" => Some(NumericKind::I8),
        "u8" => Some(NumericKind::U8),
        "i16" => Some(NumericKind::I16),
        "u16" => Some(NumericKind::U16),
        "i64" => Some(NumericKind::I64),
        "i128" => Some(NumericKind::I128),
        "isize" => Some(NumericKind::ISize),
        "u64" => Some(NumericKind::U64),
        "u128" => Some(NumericKind::U128),
        "usize" => Some(NumericKind::USize),
        "u32" => Some(NumericKind::U32),
        "i32" => Some(NumericKind::I32),
        _ => None,
    }
}

fn nonzero_numeric_kind(type_path: &TypePath) -> Option<NumericKind> {
    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    let kind = match segment.ident.to_string().as_str() {
        "NonZeroI8" => NumericKind::NonZeroI8,
        "NonZeroI16" => NumericKind::NonZeroI16,
        "NonZeroI32" => NumericKind::NonZeroI32,
        "NonZeroI64" => NumericKind::NonZeroI64,
        "NonZeroI128" => NumericKind::NonZeroI128,
        "NonZeroIsize" => NumericKind::NonZeroISize,
        "NonZeroU8" => NumericKind::NonZeroU8,
        "NonZeroU16" => NumericKind::NonZeroU16,
        "NonZeroU32" => NumericKind::NonZeroU32,
        "NonZeroU64" => NumericKind::NonZeroU64,
        "NonZeroU128" => NumericKind::NonZeroU128,
        "NonZeroUsize" => NumericKind::NonZeroUSize,
        _ => return None,
    };
    let leaf = segment.ident.to_string();
    let leaf = leaf.as_str();
    if path_is_exact_no_args(type_path, &[leaf])
        || path_is_exact_no_args(type_path, &["std", "num", leaf])
        || path_is_exact_no_args(type_path, &["core", "num", leaf])
    {
        Some(kind)
    } else {
        None
    }
}

fn is_string_type(type_path: &TypePath) -> bool {
    path_is_exact_no_args(type_path, &["String"])
        || path_is_exact_no_args(type_path, &["std", "string", "String"])
        || path_is_exact_no_args(type_path, &["alloc", "string", "String"])
}

fn is_decimal_type(type_path: &TypePath) -> bool {
    path_is_exact_no_args(type_path, &["Decimal"])
        || path_is_exact_no_args(type_path, &["rust_decimal", "Decimal"])
}

fn is_chrono_no_args_type(type_path: &TypePath, leaf: &str) -> bool {
    path_is_exact_no_args(type_path, &[leaf]) || path_is_exact_no_args(type_path, &["chrono", leaf])
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
    let leaf = match type_path.path.segments.last() {
        Some(segment)
            if matches!(segment.arguments, PathArguments::None)
                && (segment.ident == "Duration" || segment.ident == "TimeDelta") =>
        {
            segment.ident.to_string()
        }
        _ => return false,
    };
    let leaf = leaf.as_str();
    path_is_exact_no_args(type_path, &[leaf]) || path_is_exact_no_args(type_path, &["chrono", leaf])
}

fn wrapper_path_matches(type_path: &TypePath, bare: &str, qualified: &[&str]) -> bool {
    path_is_exact_with_leaf_args(type_path, &[bare])
        || path_is_exact_with_leaf_args(type_path, qualified)
}

fn extract_inner_type<'a>(ty: &'a Type, wrapper: &str, qualified: &[&str]) -> Option<&'a Type> {
    if let Type::Path(type_path) = ty
        && wrapper_path_matches(type_path, wrapper, qualified)
        && let Some(segment) = type_path.path.segments.last()
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
    {
        return Some(inner_ty);
    }
    None
}

/// Outcome of peeling a `Cow<'a, T>` layer. Cow's first generic argument is
/// the lifetime, not the inner type, so the standard `extract_inner_type`
/// helper (which keys on `args.first()`) cannot be used. Sized inner types
/// rebind and continue the transparent smart-pointer peel; unsized `str` and
/// `[T]` stay as semantic Cow leaves for parser-level domain decisions.
enum CowPeel<'a> {
    /// `Cow<'a, OwnedT>` where `OwnedT: Sized` — rebind to `OwnedT`.
    Rebind(&'a Type),
    /// `Cow<'a, str>` or `Cow<'a, [T]>` — keep the Cow as the analyzed leaf
    /// so later parser/codegen stages can apply domain-specific semantics.
    KeepAsSemanticBase,
}

/// Peel a `Cow<'a, T>` layer if the type is one. Only bare `Cow` and the
/// canonical std/alloc paths are recognized; user paths such as
/// `domain::Cow<T>` remain custom structs.
fn peel_cow(ty: &Type) -> Option<CowPeel<'_>> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    if !is_cow_path(type_path) {
        return None;
    }
    let inner_ty = cow_inner_type(type_path)?;
    if is_bare_str_type(inner_ty) {
        return Some(CowPeel::KeepAsSemanticBase);
    }
    if matches!(inner_ty, Type::Slice(_)) {
        return Some(CowPeel::KeepAsSemanticBase);
    }
    Some(CowPeel::Rebind(inner_ty))
}

fn is_cow_path(type_path: &TypePath) -> bool {
    path_is_exact_with_leaf_args(type_path, &["Cow"])
        || path_is_exact_with_leaf_args(type_path, &["std", "borrow", "Cow"])
        || path_is_exact_with_leaf_args(type_path, &["alloc", "borrow", "Cow"])
}

fn cow_inner_type(type_path: &TypePath) -> Option<&Type> {
    if !is_cow_path(type_path) {
        return None;
    }
    let segment = type_path.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    // Cow takes `<'a, T>` — find the first `Type` argument (skipping the
    // leading lifetime). Defensive against generic-syntax variation: any
    // GenericArgument::Type wins regardless of position.
    args.args.iter().find_map(|a| match a {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

fn is_bare_str_type(ty: &Type) -> bool {
    if let Type::Path(inner_path) = ty
        && inner_path.qself.is_none()
        && inner_path.path.segments.len() == 1
        && let Some(seg) = inner_path.path.segments.last()
    {
        return seg.ident == "str" && matches!(seg.arguments, PathArguments::None);
    }
    false
}

fn is_u8_type(ty: &Type) -> bool {
    if let Type::Path(inner_path) = ty
        && inner_path.qself.is_none()
        && inner_path.path.segments.len() == 1
        && let Some(seg) = inner_path.path.segments.last()
    {
        return seg.ident == "u8" && matches!(seg.arguments, PathArguments::None);
    }
    false
}

fn borrowed_reference_base(reference: &syn::TypeReference) -> Option<AnalyzedBase> {
    let inner_ty = reference.elem.as_ref();
    if is_bare_str_type(inner_ty) {
        return Some(AnalyzedBase::BorrowedStr);
    }
    if let Type::Slice(slice) = inner_ty {
        if is_u8_type(&slice.elem) {
            Some(AnalyzedBase::BorrowedBytes)
        } else {
            Some(AnalyzedBase::BorrowedSlice)
        }
    } else {
        None
    }
}

fn analyze_cow_base(type_path: &TypePath) -> Option<AnalyzedBase> {
    let inner_ty = cow_inner_type(type_path)?;
    if is_bare_str_type(inner_ty) {
        return Some(AnalyzedBase::CowStr);
    }
    if let Type::Slice(slice) = inner_ty {
        if is_u8_type(&slice.elem) {
            Some(AnalyzedBase::CowBytes)
        } else {
            Some(AnalyzedBase::CowSlice)
        }
    } else {
        None
    }
}

/// Detect a `chrono::DateTime<Tz>` field as either bare `DateTime<Tz>` or
/// canonical `chrono::DateTime<Tz>`. Qualified custom paths such as
/// `domain::DateTime<T>` remain user structs.
fn is_chrono_datetime(type_path: &TypePath) -> bool {
    if !path_is_exact_with_leaf_args(type_path, &["DateTime"])
        && !path_is_exact_with_leaf_args(type_path, &["chrono", "DateTime"])
    {
        return false;
    }
    let Some(segment) = type_path.path.segments.last() else {
        return false;
    };
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return false;
    };
    args.args
        .iter()
        .any(|arg| matches!(arg, GenericArgument::Type(_)))
}
