//! Runtime behind `#[derive(ToDataFrame)]`.
//!
//! Every supported field type names a [`ColumnBuilder`] through [`Columnar`].
//! A builder owns the buffers for one or more output columns, accepts one
//! value at a time, and hands finished columns out in schema order. Wrappers
//! such as `Option` and `Vec` wrap the builder of the type they contain, so
//! nested shapes are resolved by the compiler through trait impls.
//!
//! [`ToDataFrame`] marks the types that are whole rows (derived structs). It
//! adds the conversions users call; all of them run through one encoder.

use polars::prelude::{Column, DataFrame, DataType, PlSmallStr, PolarsResult};

/// The validity bitmap handed to [`ColumnBuilder::finish`], re-exported so a
/// custom builder does not need its own `polars-arrow` dependency.
pub use polars_arrow::bitmap::Bitmap;

pub mod builders;
mod decimal;
mod frame;

pub use decimal::Decimal128Encode;

/// Output schema fragment: column names and logical dtypes, in column order.
pub type Fields = Vec<(PlSmallStr, DataType)>;

/// Accumulates values of type `T` into one or more columns.
///
/// Implementations must add exactly one slot to every column they own on each
/// [`push`](ColumnBuilder::push) and each
/// [`push_null`](ColumnBuilder::push_null). Null tracking belongs to the
/// nearest enclosing [`builders::OptionBuilder`], which hands a validity
/// bitmap down at [`finish`](ColumnBuilder::finish).
#[diagnostic::on_unimplemented(
    message = "`{T}` cannot be encoded with `{Self}`",
    label = "this encoding does not apply to `{T}`",
    note = "the encoding chosen for this field (by a `#[df_derive(...)]` attribute or `with = \"...\"`) does not support the field's type"
)]
pub trait ColumnBuilder<T: ?Sized>: Sized {
    /// Appends this builder's columns to `out`. `name` is the dotted path of
    /// the field being described; leaves use it verbatim.
    fn schema(name: PlSmallStr, out: &mut Fields);

    /// How many list-length counters [`measure`](ColumnBuilder::measure)
    /// fills in: one per list nested anywhere inside this builder.
    const LISTS: usize = 0;

    /// Creates a builder sized for `rows` rows.
    fn with_capacity(rows: usize) -> Self;

    /// Adds the lengths of the lists inside `value` to `counts`, which has
    /// [`LISTS`](ColumnBuilder::LISTS) slots.
    ///
    /// When rows come from a slice they are measured first, so that list
    /// buffers are allocated once at their final size. A builder without
    /// nested lists keeps the default, which counts nothing.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn measure(value: &T, counts: &mut [usize]) {
        let _ = (value, counts);
    }

    /// Creates a builder for `rows` rows whose lists hold `counts` elements
    /// in total, as gathered by [`measure`](ColumnBuilder::measure).
    fn with_capacities(rows: usize, counts: &[usize]) -> Self {
        let _ = counts;
        Self::with_capacity(rows)
    }

    /// Appends one value.
    ///
    /// # Errors
    /// Returns an error when the value cannot be represented in the column.
    fn push(&mut self, value: &T) -> PolarsResult<()>;

    /// Appends one placeholder slot that an enclosing option records as null.
    fn push_null(&mut self);

    /// Appends every value of a slice.
    ///
    /// # Errors
    /// Returns the first error produced by [`ColumnBuilder::push`].
    #[inline]
    fn extend(&mut self, values: &[T]) -> PolarsResult<()>
    where
        T: Sized,
    {
        for value in values {
            self.push(value)?;
        }
        Ok(())
    }

    /// Consumes the builder and appends its columns to `out`, named exactly
    /// as [`ColumnBuilder::schema`] names them for the same `name`.
    /// `validity`, when present, has one bit per pushed slot and applies to
    /// every column.
    ///
    /// # Errors
    /// Returns an error if a column cannot be assembled.
    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()>;
}

/// A type that can be stored in `DataFrame` columns.
///
/// Implemented for the supported leaf types and wrappers, and by
/// `#[derive(ToDataFrame)]` for structs. A field of any `Columnar` type works
/// in a derived struct without attributes.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be stored in a DataFrame column",
    label = "no column encoding for `{Self}`",
    note = "derive `ToDataFrame` on the type, or choose an encoding on the field with an attribute such as `#[df_derive(as_str)]`, `#[df_derive(as_string)]`, or `#[df_derive(as_binary)]`"
)]
pub trait Columnar {
    /// The builder used when no field-level encoding is chosen.
    type Builder: ColumnBuilder<Self>;
}

/// A type whose columns are named after its own fields.
///
/// Derived structs and tuples are records, and so is any `Option`, `Vec` or
/// pointer around one. A leaf such as `f64` is not: it takes the name of the
/// field that holds it. `#[df_derive(flatten)]` requires a record, because
/// flattening drops the field's own name.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no named columns of its own",
    label = "not a struct or tuple of columns",
    note = "`#[df_derive(flatten)]` and `ToDataFrame` need a type whose columns are named by its fields"
)]
pub trait Record: Columnar {}

/// A row type: each value becomes one `DataFrame` row.
///
/// `#[derive(ToDataFrame)]` implements this. Every method has a default that
/// runs the type's [`Columnar`] builder.
pub trait ToDataFrame: Record + Sized {
    /// Converts this value into a one-row `DataFrame`.
    ///
    /// # Errors
    /// Returns an error if a field cannot be encoded.
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        frame::encode_slice(::core::slice::from_ref(self))
    }

    /// Converts borrowed rows into a `DataFrame` in one pass.
    ///
    /// # Errors
    /// Returns an error naming the row and field that could not be encoded.
    fn rows_to_dataframe<'a, I>(rows: I) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        I: IntoIterator<Item = &'a Self>,
    {
        frame::encode(rows)
    }

    /// Returns a zero-row `DataFrame` with this type's schema.
    ///
    /// # Errors
    /// Returns an error if the schema has duplicate column names.
    fn empty_dataframe() -> PolarsResult<DataFrame> {
        frame::encode_slice::<Self>(&[])
    }

    /// Returns this type's column names and dtypes, in column order.
    ///
    /// # Errors
    /// Returns an error if the schema has duplicate column names.
    fn schema() -> PolarsResult<Vec<(String, DataType)>> {
        frame::schema::<Self>()
    }
}

/// Adds `.to_dataframe()` to slices (and `Vec` through auto-deref).
pub trait ToDataFrameVec {
    /// Converts every element into one `DataFrame` row.
    ///
    /// # Errors
    /// Returns an error naming the row and field that could not be encoded.
    fn to_dataframe(&self) -> PolarsResult<DataFrame>;
}

impl<T: ToDataFrame> ToDataFrameVec for [T] {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        frame::encode_slice(self)
    }
}

impl Record for () {}
impl ToDataFrame for () {}

/// Items generated code refers to. Not part of the public API.
#[doc(hidden)]
pub mod __private {
    pub use super::frame::{child, conversion_error, in_field};

    /// Compile-time check that a flattened field is a record.
    pub const fn assert_record<T: ?Sized + super::Record>() {}
    pub use polars;
    pub use polars_arrow;
}
