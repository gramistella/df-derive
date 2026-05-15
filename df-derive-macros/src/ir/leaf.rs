use syn::{Ident, Type};

use super::TupleElement;

/// Datetime time unit chosen via `#[df_derive(time_unit = "ms"|"us"|"ns")]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DateTimeUnit {
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

/// Source of a `Duration` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurationSource {
    Std,
    Chrono,
}

/// Numeric primitive kind, including widened platform-sized integers and
/// `std::num::NonZero*` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericKind {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    /// Widens to `i64`.
    ISize,
    /// Widens to `u64`.
    USize,
    NonZeroI8,
    NonZeroI16,
    NonZeroI32,
    NonZeroI64,
    NonZeroI128,
    /// Widens to `i64`.
    NonZeroISize,
    NonZeroU8,
    NonZeroU16,
    NonZeroU32,
    NonZeroU64,
    NonZeroU128,
    /// Widens to `u64`.
    NonZeroUSize,
}

impl NumericKind {
    pub const fn storage_kind(self) -> StorageNumericKind {
        match self {
            Self::I8 | Self::NonZeroI8 => StorageNumericKind::I8,
            Self::I16 | Self::NonZeroI16 => StorageNumericKind::I16,
            Self::I32 | Self::NonZeroI32 => StorageNumericKind::I32,
            Self::I64 | Self::NonZeroI64 => StorageNumericKind::I64,
            Self::I128 | Self::NonZeroI128 => StorageNumericKind::I128,
            Self::U8 | Self::NonZeroU8 => StorageNumericKind::U8,
            Self::U16 | Self::NonZeroU16 => StorageNumericKind::U16,
            Self::U32 | Self::NonZeroU32 => StorageNumericKind::U32,
            Self::U64 | Self::NonZeroU64 => StorageNumericKind::U64,
            Self::U128 | Self::NonZeroU128 => StorageNumericKind::U128,
            Self::F32 => StorageNumericKind::F32,
            Self::F64 => StorageNumericKind::F64,
            Self::ISize | Self::NonZeroISize => StorageNumericKind::ISize,
            Self::USize | Self::NonZeroUSize => StorageNumericKind::USize,
        }
    }

    pub const fn is_widened(self) -> bool {
        matches!(
            self.storage_kind(),
            StorageNumericKind::ISize | StorageNumericKind::USize
        )
    }

    pub const fn is_nonzero(self) -> bool {
        matches!(
            self,
            Self::NonZeroI8
                | Self::NonZeroI16
                | Self::NonZeroI32
                | Self::NonZeroI64
                | Self::NonZeroI128
                | Self::NonZeroISize
                | Self::NonZeroU8
                | Self::NonZeroU16
                | Self::NonZeroU32
                | Self::NonZeroU64
                | Self::NonZeroU128
                | Self::NonZeroUSize
        )
    }
}

/// Primitive numeric storage lane after non-zero wrappers have been erased.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageNumericKind {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    ISize,
    USize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StringyBase {
    String,
    BorrowedStr,
    CowStr,
    Struct(Type),
    Generic(Ident),
}

impl StringyBase {
    pub const fn is_string(&self) -> bool {
        matches!(self, Self::String)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DisplayBase {
    Inherent,
    Struct(Type),
    Generic(Ident),
}

/// Per-leaf semantic shape after type analysis and field overrides have been
/// lowered into the encoder vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeafSpec {
    Numeric(NumericKind),
    String,
    Bool,
    DateTime(DateTimeUnit),
    NaiveDateTime(DateTimeUnit),
    NaiveDate,
    NaiveTime,
    Duration {
        unit: DateTimeUnit,
        source: DurationSource,
    },
    Decimal {
        precision: u8,
        scale: u8,
    },
    AsString(DisplayBase),
    AsStr(StringyBase),
    Binary,
    Struct(Type),
    Generic(Ident),
    Tuple(Vec<TupleElement>),
}

/// Borrowed view of a single-column primitive leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveLeaf<'a> {
    Numeric(NumericKind),
    String,
    Bool,
    Binary,
    DateTime(DateTimeUnit),
    NaiveDateTime(DateTimeUnit),
    NaiveDate,
    NaiveTime,
    Duration {
        unit: DateTimeUnit,
        source: DurationSource,
    },
    Decimal {
        precision: u8,
        scale: u8,
    },
    AsString,
    AsStr(&'a StringyBase),
}

impl PrimitiveLeaf<'_> {
    pub const fn is_copy(self) -> bool {
        matches!(
            self,
            Self::Numeric(_)
                | Self::Bool
                | Self::NaiveDate
                | Self::NaiveTime
                | Self::NaiveDateTime(_)
                | Self::Duration { .. }
        )
    }
}

/// Borrowed view of a multi-column nested leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestedLeaf<'a> {
    Struct(&'a Type),
    Generic(&'a Ident),
}

/// Lossless borrowed route for a [`LeafSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafRoute<'a> {
    Primitive(PrimitiveLeaf<'a>),
    Nested(NestedLeaf<'a>),
    Tuple(&'a [TupleElement]),
}

impl LeafSpec {
    pub const fn route(&self) -> LeafRoute<'_> {
        match self {
            Self::Numeric(kind) => LeafRoute::Primitive(PrimitiveLeaf::Numeric(*kind)),
            Self::String => LeafRoute::Primitive(PrimitiveLeaf::String),
            Self::Bool => LeafRoute::Primitive(PrimitiveLeaf::Bool),
            Self::DateTime(unit) => LeafRoute::Primitive(PrimitiveLeaf::DateTime(*unit)),
            Self::NaiveDateTime(unit) => LeafRoute::Primitive(PrimitiveLeaf::NaiveDateTime(*unit)),
            Self::NaiveDate => LeafRoute::Primitive(PrimitiveLeaf::NaiveDate),
            Self::NaiveTime => LeafRoute::Primitive(PrimitiveLeaf::NaiveTime),
            Self::Duration { unit, source } => LeafRoute::Primitive(PrimitiveLeaf::Duration {
                unit: *unit,
                source: *source,
            }),
            Self::Decimal { precision, scale } => LeafRoute::Primitive(PrimitiveLeaf::Decimal {
                precision: *precision,
                scale: *scale,
            }),
            Self::AsString(_) => LeafRoute::Primitive(PrimitiveLeaf::AsString),
            Self::AsStr(stringy) => LeafRoute::Primitive(PrimitiveLeaf::AsStr(stringy)),
            Self::Binary => LeafRoute::Primitive(PrimitiveLeaf::Binary),
            Self::Struct(ty) => LeafRoute::Nested(NestedLeaf::Struct(ty)),
            Self::Generic(ident) => LeafRoute::Nested(NestedLeaf::Generic(ident)),
            Self::Tuple(elements) => LeafRoute::Tuple(elements.as_slice()),
        }
    }

    #[allow(dead_code)]
    pub const fn is_tuple(&self) -> bool {
        matches!(self, Self::Tuple(_))
    }
}
