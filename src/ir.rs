use syn::Ident;

/// Top-level IR for a struct targeted by the derive.
#[derive(Clone)]
pub struct StructIR {
    /// The identifier of the struct being derived
    pub name: Ident,
    /// The generics declared on the struct (empty when no generics are used)
    pub generics: syn::Generics,
    /// The fields of the struct in declaration order
    pub fields: Vec<FieldIR>,
}

/// IR for a single field of a struct
#[derive(Clone)]
pub struct FieldIR {
    /// Field name as declared on the struct
    pub name: Ident,
    /// Field index for tuple structs (None for named fields)
    pub field_index: Option<usize>,
    /// Wrappers applied from outermost to innermost, e.g. [Option, Vec]
    pub wrappers: Vec<Wrapper>,
    /// The base Rust type of the field (primitive or user-defined struct)
    pub base_type: BaseType,
    /// Optional transform to apply when materializing values
    pub transform: Option<PrimitiveTransform>,
    /// The original `syn::Type` of the field. Preserved so codegen can splice
    /// it into trait-bound asserts (e.g. `T: AsRef<str>`) with the user's
    /// source span, putting compiler errors at the field declaration rather
    /// than deep in macro expansion.
    pub field_ty: syn::Type,
}

/// Optional transformation applied to primitive values during codegen
#[derive(Clone)]
pub enum PrimitiveTransform {
    /// Convert `chrono::DateTime<Utc>` to an epoch integer (i64) at the chosen
    /// `Datetime` precision. The chosen unit determines both the chrono call
    /// used to produce the i64 (`timestamp_millis` / `timestamp_micros` /
    /// `timestamp_nanos_opt`) and the `DataType::Datetime(...)` cast target.
    DateTimeToInt(DateTimeUnit),
    /// Convert `rust_decimal::Decimal` to its `i128` mantissa rescaled to the
    /// schema `scale`, then build an `Int128Chunked` cast directly to
    /// `Decimal(precision, scale)`. `precision` is bounded by the Polars
    /// invariant `1 <= precision <= 38`. Rescale matches the historical
    /// `to_string + parse` round-trip through polars's `str_to_dec128`:
    /// scale-up multiplies the mantissa (and surfaces overflow as a polars
    /// `ComputeError`); scale-down rounds the magnitude using
    /// round-half-to-even (banker's rounding) and re-applies the sign.
    DecimalToInt128 { precision: u8, scale: u8 },
    /// Convert any value to `String` using `ToString`
    ToString,
    /// Borrow `&str` via `<T as AsRef<str>>::as_ref` for the duration of the
    /// columnar populator pass. Zero-allocation per row.
    AsStr,
}

/// Datetime time unit chosen via `#[df_derive(time_unit = "ms"|"us"|"ns")]`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DateTimeUnit {
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

/// The base Rust type (primitive or user-defined struct)
#[derive(Clone)]
pub enum BaseType {
    F64,
    F32,
    I64,
    U64,
    I32,
    U32,
    I16,
    U16,
    I8,
    U8,
    Bool,
    String,
    ISize,
    USize,
    DateTimeUtc,
    Decimal,
    /// A concrete user-defined struct type. The first element is the bare
    /// ident (last path segment); the second carries any angle-bracketed
    /// generic arguments declared at the field's use site, e.g. `<M>` in
    /// `Vec<Foo<M>>`. `None` means the type was used without arguments.
    Struct(Ident, Option<syn::AngleBracketedGenericArguments>),
    /// A generic type parameter declared on the enclosing struct
    Generic(Ident),
}

/// Wrapper layers applied around the base type
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Wrapper {
    Option,
    Vec,
}

/// True if any wrapper layer is `Vec<…>`.
pub fn has_vec(wrappers: &[Wrapper]) -> bool {
    wrappers.iter().any(|w| matches!(w, Wrapper::Vec))
}

/// Number of `Vec<…>` layers in the wrapper stack.
pub fn vec_count(wrappers: &[Wrapper]) -> usize {
    wrappers
        .iter()
        .filter(|w| matches!(w, Wrapper::Vec))
        .count()
}
