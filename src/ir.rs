use syn::Ident;

/// Top-level IR for a struct targeted by the derive.
#[derive(Clone)]
pub struct StructIR {
    /// The identifier of the struct being derived
    pub name: Ident,
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
}

/// Optional transformation applied to primitive values during codegen
#[derive(Clone)]
pub enum PrimitiveTransform {
    /// Convert `chrono::DateTime<Utc>` to epoch milliseconds (i64)
    DateTimeToMillis,
    /// Convert `rust_decimal::Decimal` to `String`
    DecimalToString,
    /// Convert any value to `String` using `ToString`
    ToString,
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
    Struct(Ident),
}

/// Wrapper layers applied around the base type
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Wrapper {
    Option,
    Vec,
}
