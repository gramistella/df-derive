//! Runtime behind `#[derive(ToDataFrame)]`.
//!
//! Every supported field type names a [`ColumnBuilder`] through [`Columnar`].
//! A builder owns the buffers of one or more output columns and hands the
//! finished columns out in schema order. Wrappers such as `Option` and `Vec`
//! wrap the builder of the type they contain, so nested shapes are resolved
//! by the compiler through trait impls.
//!
//! [`ToDataFrame`] marks the types that are whole rows (derived structs). It
//! adds the conversions users call; all of them run through one encoder.

use polars::prelude::{Column, DataFrame, DataType, PlSmallStr, PolarsResult};

/// The validity bitmap handed to [`ColumnBuilder::finish`], re-exported so a
/// custom builder does not need its own `polars-arrow` dependency.
pub use polars_arrow::bitmap::Bitmap;

pub mod builders;
#[cfg(feature = "decimal")]
mod decimal;
mod frame;
#[cfg(feature = "categorical")]
mod variants;

#[cfg(feature = "decimal")]
pub use decimal::Decimal128Encode;
#[cfg(feature = "categorical")]
pub use variants::Variants;

/// Output schema fragment: column names and logical dtypes, in column order.
pub type Fields = Vec<(PlSmallStr, DataType)>;

/// Accumulates values of type `T` into one or more columns.
///
/// A builder is filled one *slot* at a time. A slot is `Some(value)`, or
/// `None` for a position that holds no value: the builder appends a
/// placeholder there, and the nearest enclosing [`builders::OptionBuilder`]
/// records the null and hands a validity bitmap down at
/// [`finish`](ColumnBuilder::finish). Every column a builder owns gets exactly
/// one entry per slot.
///
/// The life of a builder:
///
/// 1. [`new`](ColumnBuilder::new) creates it empty.
/// 2. [`measure`](ColumnBuilder::measure) shows it the slots that are coming
///    and [`reserve`](ColumnBuilder::reserve) allocates for them, so that
///    every buffer is allocated once, at its final size.
/// 3. [`push`](ColumnBuilder::push) appends one slot and
///    [`extend`](ColumnBuilder::extend) appends many.
/// 4. [`finish`](ColumnBuilder::finish) hands the columns over.
///
/// Step 2 is optional: a builder that was not told what is coming grows as it
/// fills.
///
/// `schema`, `new`, `reserve`, `push` and `finish` are required and describe
/// an encoding completely. `measure`, `extend` and the three constants have
/// defaults that are always correct; builders override them to be exact about
/// memory and fast. Whether a value can be appended must not depend on the
/// slots before it.
#[diagnostic::on_unimplemented(
    message = "`{T}` cannot be encoded with `{Self}`",
    label = "this encoding does not apply to `{T}`",
    note = "the encoding chosen for this field (by a `#[df_derive(...)]` attribute or `with = \"...\"`) does not support the field's type"
)]
pub trait ColumnBuilder<T: ?Sized>: Sized {
    /// Whether a row's builder should feed this one a row at a time, with
    /// [`push`](ColumnBuilder::push), instead of a block at a time with
    /// [`extend`](ColumnBuilder::extend).
    ///
    /// A block at a time suits a builder that turns a run of values into one
    /// tight loop over a buffer, and it is the default. A builder that reaches
    /// each value through a pointer and appends it with a call, as those that
    /// borrow a string or bytes do, gains nothing from a run of its own. The
    /// builders of a row that set this are fed together, in one walk over the
    /// rows of the block.
    ///
    /// It is a hint. The columns come out the same either way.
    const ROW_AT_A_TIME: bool = false;

    /// How many columns [`finish`](ColumnBuilder::finish) appends.
    ///
    /// The builder of a projected type pushes a row of few columns straight
    /// from the conversion, and makes a block of wider rows before it appends
    /// them in runs.
    ///
    /// It is a hint. The columns come out the same either way.
    const COLUMNS: usize = 1;

    /// Whether [`measure`](ColumnBuilder::measure) counts anything.
    ///
    /// The builders that hold lists set this, and wrappers pass it on. The
    /// builder of a projected type has to make a row to show it, and does
    /// that only if this says the row's builder would count something.
    ///
    /// It is a hint. The columns come out the same either way.
    const MEASURES: bool = false;

    /// Appends this builder's columns to `out`. `name` is the dotted path of
    /// the field being described; leaves use it verbatim.
    ///
    /// Naming a column cannot fail. A builder that may be unable to make the
    /// type of its column names `DataType::Unknown` for it when that
    /// happens, and returns the reason from
    /// [`finish`](ColumnBuilder::finish).
    fn schema(name: PlSmallStr, out: &mut Fields);

    /// Creates an empty builder.
    fn new() -> Self;

    /// Looks at slots that are about to be appended, ahead of
    /// [`reserve`](ColumnBuilder::reserve).
    ///
    /// A builder that holds lists adds up their lengths here: the number of
    /// slots does not say how large the buffer behind the lists must be. It
    /// also sets [`MEASURES`](ColumnBuilder::MEASURES). The default does
    /// nothing, and wrappers pass the slots on to the builders they wrap.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn measure<'a, I>(&mut self, values: I)
    where
        T: 'a,
        I: Iterator<Item = Option<&'a T>> + Clone,
    {
        let _ = values;
    }

    /// Makes room for `additional` more slots, and for everything
    /// [`measure`](ColumnBuilder::measure) has counted since the last call.
    fn reserve(&mut self, additional: usize);

    /// Appends one slot.
    ///
    /// # Errors
    /// Returns an error when the value cannot be represented in the column.
    /// A builder made of other builders says which part of the value failed
    /// by prefixing the error with the field's name. Its columns may then
    /// differ in length: a builder that returned an error must not be used
    /// again.
    fn push(&mut self, value: Option<&T>) -> PolarsResult<()>;

    /// Appends one slot per item, with the same result as pushing them in
    /// order.
    ///
    /// This is how rows are encoded: a block at a time, each field's builder
    /// getting an iterator over that field in every row of the block, so a
    /// column is filled by one tight loop. The iterator is `Clone` because a
    /// builder with several buffers walks it once per buffer. Builders that
    /// set [`ROW_AT_A_TIME`](ColumnBuilder::ROW_AT_A_TIME) are pushed to
    /// instead.
    ///
    /// # Errors
    /// Returns an error when any of the values cannot be represented. Which
    /// of them the error is about is unspecified: a caller that needs to know
    /// pushes the values one at a time. A builder that returned an error must
    /// not be used again.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        T: 'a,
        I: Iterator<Item = Option<&'a T>> + Clone,
    {
        for value in values {
            self.push(value)?;
        }
        Ok(())
    }

    /// Consumes the builder and appends its columns to `out`, named exactly
    /// as [`schema`](ColumnBuilder::schema) names them for the same `name`.
    /// `validity`, when present, has one bit per slot and applies to every
    /// column.
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
///
/// A builder borrows nothing from the values it reads, so its type must not
/// be tied to their lifetimes: the builder that `Name<'a>` names has to
/// encode a `Name<'b>` too. A derived type that holds a `Name<'a>` in a field
/// relies on this. It keeps one builder for the field, named without the
/// lifetime, and feeds it values of any lifetime.
///
/// - Give the builder of a type with lifetime parameters none of its own:
///   `impl<'a> ColumnBuilder<Name<'a>> for NameBuilder`.
/// - Make the builder of a generic wrapper generic over the builder of what
///   it wraps, not over what it wraps:
///   `impl<T, B: ColumnBuilder<T>> ColumnBuilder<Wrapper<T>> for WrapperBuilder<B>`,
///   named by `Wrapper<T>` as `WrapperBuilder<T::Builder>`. A lifetime inside
///   `T` then does not reach the builder's type. The builders in
///   [`builders`] are written this way, and so are derived ones.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be stored in a DataFrame column",
    label = "no column encoding for `{Self}`",
    note = "derive `ToDataFrame` on the type, implement `Columnar` for it, or choose an encoding on the field with an attribute such as `#[df_derive(as_str)]`, `#[df_derive(as_string)]`, or `#[df_derive(as_binary)]`",
    note = "the key type of a map needs one of the first two: a field attribute applies to the map's values"
)]
// Without the feature that covers it, a type looks like any other type with
// no encoding. The error then says which features are off.
#[cfg_attr(
    not(feature = "chrono"),
    diagnostic::on_unimplemented(
        note = "df-derive's `chrono` feature is off: turn it on if `{Self}` is a `chrono` type"
    )
)]
#[cfg_attr(
    not(feature = "rust_decimal"),
    diagnostic::on_unimplemented(
        note = "df-derive's `rust_decimal` feature is off: turn it on if `{Self}` is `rust_decimal::Decimal`"
    )
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
/// flattening drops the field's own name, and so does `via`, because the type
/// it names is the row a projected type is encoded as.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no named columns of its own",
    label = "not a struct or tuple of columns",
    note = "`#[df_derive(flatten)]`, `via` and `ToDataFrame` need a type whose columns are named by its fields"
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
        Self::__encode(::core::slice::from_ref(self))
    }

    /// Converts borrowed rows into a `DataFrame`.
    ///
    /// Use this when the rows are not in one slice: scattered references, a
    /// filtered view, a map's values. A slice or `Vec` of rows converts
    /// directly with [`ToDataFrameVec::to_dataframe`].
    ///
    /// The references are gathered into a vector first. Their encoder is
    /// compiled in the crate that calls this: a second derive-written one,
    /// beside the encoder for a slice, would nearly double what every derived
    /// type costs to compile.
    ///
    /// # Errors
    /// Returns an error naming the row and field that could not be encoded.
    fn rows_to_dataframe<'a, I>(rows: I) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        I: IntoIterator<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.into_iter().collect();
        frame::encode::<Self, &Self>(&rows)
    }

    /// Returns a zero-row `DataFrame` with this type's schema.
    ///
    /// # Errors
    /// Returns an error if the schema has duplicate column names.
    fn empty_dataframe() -> PolarsResult<DataFrame> {
        Self::__encode(&[])
    }

    /// Returns this type's column names and dtypes, in column order.
    ///
    /// # Errors
    /// Returns an error if the schema has duplicate column names, or if the
    /// type of a column cannot be made. No frame of this type can be built
    /// then either, and the error is the one building it returns.
    fn schema() -> PolarsResult<Vec<(String, DataType)>> {
        frame::schema::<Self>(Self::empty_dataframe)
    }

    /// Encodes a slice of rows: where the conversions above end up.
    ///
    /// The derive writes this method out instead of inheriting it: the
    /// encoder is then compiled once, in the crate that defines the row type,
    /// not in every crate that converts one.
    #[doc(hidden)]
    fn __encode(rows: &[Self]) -> PolarsResult<DataFrame> {
        frame::encode::<Self, Self>(rows)
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
        T::__encode(self)
    }
}

impl Record for () {}
impl ToDataFrame for () {}

/// Items generated code refers to. Not part of the public API.
#[doc(hidden)]
pub mod __private {
    pub use polars;
    pub use polars_arrow;

    pub use super::frame::{
        child, conversion_error, extend_through, in_field, measure_through, measures_through,
        push_through,
    };

    /// Compile-time check that a flattened field is a record.
    pub const fn assert_record<T: ?Sized + super::Record>() {}

    /// The body of a derived `ToDataFrame::__encode`.
    ///
    /// # Errors
    /// Returns an error naming the row and field that could not be encoded.
    pub fn encode<T: super::Columnar>(
        rows: &[T],
    ) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
        super::frame::encode::<T, T>(rows)
    }
}
