//! Builder-based column encoding.
//!
//! Every supported field type names a [`ColumnBuilder`] through [`Encode`].
//! A builder owns the buffers for one or more output columns, accepts one row
//! at a time, and hands finished columns out in schema order. Wrappers
//! (`Option`, `Vec`) compose by wrapping the builder of the type they contain,
//! so nested shapes are resolved by the compiler instead of by the derive.
//!
//! Invariant every builder upholds: each `push` and each `push_null` adds
//! exactly one slot to every column the builder owns. Validity is tracked by
//! the nearest enclosing [`OptionBuilder`] and handed down at `finish`.
//!
//! The hot `push` paths are `#[inline(always)]` on purpose: a row loop only
//! matches hand-written column code when every field push is inlined into it.

#![allow(clippy::inline_always)]

use std::marker::PhantomData;

use polars::prelude::{
    BooleanChunked, Column, DataFrame, DataType, Int64Chunked, Int128Chunked, IntoSeries,
    PlSmallStr, PolarsError, PolarsResult, Series, StringChunked, TimeUnit,
};
use polars_arrow::array::{BooleanArray, ListArray, MutableBinaryViewArray, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::{Offsets, OffsetsBuffer};
use polars_arrow::types::NativeType;

use crate::dataframe::Decimal128Encode;

/// Output schema fragment: column names and logical dtypes, in column order.
pub type Fields = Vec<(PlSmallStr, DataType)>;

/// Accumulates values of type `T` into one or more columns.
pub trait ColumnBuilder<T: ?Sized>: Sized {
    /// Appends this builder's columns to `out`. `name` is the dotted path of
    /// the field being described; leaves use it verbatim.
    fn schema(name: PlSmallStr, out: &mut Fields);

    /// Creates a builder sized for `rows` rows.
    fn with_capacity(rows: usize) -> Self;

    /// Appends one value.
    ///
    /// # Errors
    /// Returns an error when the value cannot be represented in the column.
    fn push(&mut self, value: &T) -> PolarsResult<()>;

    /// Appends one placeholder slot whose validity an enclosing
    /// [`OptionBuilder`] records as null.
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

/// A type with a default column encoding.
pub trait Encode {
    /// The builder used when no field-level override is given.
    type Builder: ColumnBuilder<Self>;
}

/// Joins a parent column path and a field name with a dot.
#[must_use]
pub fn child(parent: &PlSmallStr, field: &'static str) -> PlSmallStr {
    if parent.is_empty() {
        return PlSmallStr::from_static(field);
    }
    let mut joined = String::with_capacity(parent.len() + 1 + field.len());
    joined.push_str(parent);
    joined.push('.');
    joined.push_str(field);
    joined.into()
}

/// Returns the column names and dtypes `T` encodes to.
#[must_use]
pub fn schema<T: Encode + ?Sized>() -> Fields {
    let mut fields = Fields::new();
    <T::Builder as ColumnBuilder<T>>::schema(PlSmallStr::EMPTY, &mut fields);
    fields
}

/// Encodes rows into a `DataFrame` in one pass.
///
/// # Errors
/// Returns an error naming the row when a value cannot be encoded.
pub fn to_dataframe<'a, T, I>(rows: I) -> PolarsResult<DataFrame>
where
    T: Encode + 'a,
    I: IntoIterator<Item = &'a T>,
{
    let rows = rows.into_iter();
    let mut builder = <T::Builder as ColumnBuilder<T>>::with_capacity(rows.size_hint().0);
    let mut height = 0usize;
    for row in rows {
        if let Err(error) = builder.push(row) {
            return Err(at_row(&error, std::any::type_name::<T>(), height));
        }
        height += 1;
    }
    let mut columns = Vec::new();
    builder.finish(PlSmallStr::EMPTY, None, &mut columns)?;
    DataFrame::new(height, columns)
}

#[cold]
#[inline(never)]
fn at_row(error: &PolarsError, type_name: &str, row: usize) -> PolarsError {
    PolarsError::ComputeError(format!("{type_name}[{row}]: {error}").into())
}

// ---------------------------------------------------------------------------
// Numeric leaves
// ---------------------------------------------------------------------------

/// Native numeric types that map one-to-one onto a Polars dtype.
pub trait Primitive: NativeType {
    /// The Polars dtype of a column of this type.
    const DTYPE: DataType;

    /// Wraps a finished array as a named series of [`Primitive::DTYPE`].
    fn series(name: PlSmallStr, array: PrimitiveArray<Self>) -> Series;
}

macro_rules! primitive {
    ($($ty:ty => $dtype:ident, $chunked:ident);* $(;)?) => {$(
        impl Primitive for $ty {
            const DTYPE: DataType = DataType::$dtype;

            fn series(name: PlSmallStr, array: PrimitiveArray<Self>) -> Series {
                polars::prelude::$chunked::with_chunk(name, array).into_series()
            }
        }
        impl Encode for $ty {
            type Builder = PrimitiveBuilder<$ty>;
        }
    )*};
}

primitive! {
    i8 => Int8, Int8Chunked;
    i16 => Int16, Int16Chunked;
    i32 => Int32, Int32Chunked;
    i64 => Int64, Int64Chunked;
    i128 => Int128, Int128Chunked;
    u8 => UInt8, UInt8Chunked;
    u16 => UInt16, UInt16Chunked;
    u32 => UInt32, UInt32Chunked;
    u64 => UInt64, UInt64Chunked;
    u128 => UInt128, UInt128Chunked;
    f32 => Float32, Float32Chunked;
    f64 => Float64, Float64Chunked;
}

fn primitive_array<P: NativeType>(
    values: Vec<P>,
    validity: Option<&Bitmap>,
) -> PolarsResult<PrimitiveArray<P>> {
    PrimitiveArray::try_new(P::PRIMITIVE.into(), values.into(), validity.cloned())
}

/// Builder for a column of native numeric values.
pub struct PrimitiveBuilder<P>(Vec<P>);

impl<P: Primitive> ColumnBuilder<P> for PrimitiveBuilder<P> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, P::DTYPE));
    }

    fn with_capacity(rows: usize) -> Self {
        Self(Vec::with_capacity(rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &P) -> PolarsResult<()> {
        self.0.push(*value);
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.0.push(P::default());
    }

    #[inline(always)]
    fn extend(&mut self, values: &[P]) -> PolarsResult<()> {
        self.0.extend_from_slice(values);
        Ok(())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        out.push(P::series(name, primitive_array(self.0, validity)?).into());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Boolean leaf
// ---------------------------------------------------------------------------

/// Builder for a boolean column.
pub struct BoolBuilder(BitmapBuilder);

impl ColumnBuilder<bool> for BoolBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::Boolean));
    }

    fn with_capacity(rows: usize) -> Self {
        Self(BitmapBuilder::with_capacity(rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &bool) -> PolarsResult<()> {
        self.0.push(*value);
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.0.push(false);
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let array =
            BooleanArray::try_new(ArrowDataType::Boolean, self.0.freeze(), validity.cloned())?;
        out.push(BooleanChunked::with_chunk(name, array).into_series().into());
        Ok(())
    }
}

impl Encode for bool {
    type Builder = BoolBuilder;
}

// ---------------------------------------------------------------------------
// String leaf
// ---------------------------------------------------------------------------

/// Builder for a string column from anything that borrows as `str`.
pub struct StrBuilder(MutableBinaryViewArray<str>);

impl<T: AsRef<str> + ?Sized> ColumnBuilder<T> for StrBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::String));
    }

    fn with_capacity(rows: usize) -> Self {
        Self(MutableBinaryViewArray::with_capacity(rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &T) -> PolarsResult<()> {
        self.0.push_value_ignore_validity(value.as_ref());
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.0.push_value_ignore_validity("");
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let array = self.0.freeze().with_validity(validity.cloned());
        out.push(StringChunked::with_chunk(name, array).into_series().into());
        Ok(())
    }
}

impl Encode for String {
    type Builder = StrBuilder;
}

impl Encode for str {
    type Builder = StrBuilder;
}

// ---------------------------------------------------------------------------
// Option
// ---------------------------------------------------------------------------

/// Adds a validity bitmap on top of another builder.
pub struct OptionBuilder<B> {
    inner: B,
    validity: BitmapBuilder,
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<Option<T>> for OptionBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        B::schema(name, out);
    }

    fn with_capacity(rows: usize) -> Self {
        Self {
            inner: B::with_capacity(rows),
            validity: BitmapBuilder::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &Option<T>) -> PolarsResult<()> {
        if let Some(value) = value {
            self.inner.push(value)?;
            self.validity.push(true);
        } else {
            self.inner.push_null();
            self.validity.push(false);
        }
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.inner.push_null();
        self.validity.push(false);
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let own = self.validity.into_opt_validity();
        let combined = combine_validities_and(validity, own.as_ref());
        self.inner.finish(name, combined.as_ref(), out)
    }
}

impl<T: Encode> Encode for Option<T> {
    type Builder = OptionBuilder<T::Builder>;
}

// ---------------------------------------------------------------------------
// Vec
// ---------------------------------------------------------------------------

/// Wraps every column of another builder in a list.
pub struct ListBuilder<B> {
    inner: B,
    offsets: Offsets<i64>,
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<Vec<T>> for ListBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        let start = out.len();
        B::schema(name, out);
        for (_, dtype) in &mut out[start..] {
            let inner = std::mem::replace(dtype, DataType::Null);
            *dtype = DataType::List(Box::new(inner));
        }
    }

    fn with_capacity(rows: usize) -> Self {
        Self {
            inner: B::with_capacity(rows),
            offsets: Offsets::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &Vec<T>) -> PolarsResult<()> {
        self.inner.extend(value)?;
        self.offsets.try_push(value.len())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.offsets.extend_constant(1);
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let start = out.len();
        self.inner.finish(name, None, out)?;
        let offsets: OffsetsBuffer<i64> = self.offsets.into();
        for column in &mut out[start..] {
            let series = column.as_materialized_series().rechunk();
            let values = series.chunks()[0].clone();
            let list = ListArray::<i64>::try_new(
                ListArray::<i64>::default_datatype(values.dtype().clone()),
                offsets.clone(),
                values,
                validity.cloned(),
            )?;
            let dtype = DataType::List(Box::new(series.dtype().clone()));
            *column =
                Series::from_chunk_and_dtype(series.name().clone(), Box::new(list), &dtype)?.into();
        }
        Ok(())
    }
}

impl<T: Encode> Encode for Vec<T> {
    type Builder = ListBuilder<T::Builder>;
}

// ---------------------------------------------------------------------------
// Decimal
// ---------------------------------------------------------------------------

/// Builder for a `Decimal(P, S)` column from any [`Decimal128Encode`] value.
pub struct DecimalBuilder<const P: usize, const S: usize>(Vec<i128>);

#[cold]
#[inline(never)]
fn decimal_rescale_error(scale: usize) -> PolarsError {
    PolarsError::ComputeError(
        format!(
            "df-derive: decimal mantissa rescale to scale {scale} failed (overflow or precision loss)"
        )
        .into(),
    )
}

#[cold]
#[inline(never)]
fn decimal_precision_error(mantissa: i128, precision: usize, scale: usize) -> PolarsError {
    PolarsError::ComputeError(
        format!(
            "df-derive: decimal mantissa {mantissa} exceeds declared precision {precision} for Decimal({precision}, {scale})"
        )
        .into(),
    )
}

impl<T, const P: usize, const S: usize> ColumnBuilder<T> for DecimalBuilder<P, S>
where
    T: Decimal128Encode + ?Sized,
{
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::Decimal(P, S)));
    }

    fn with_capacity(rows: usize) -> Self {
        Self(Vec::with_capacity(rows))
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn push(&mut self, value: &T) -> PolarsResult<()> {
        let Some(mantissa) = value.try_to_i128_mantissa(S as u32) else {
            return Err(decimal_rescale_error(S));
        };
        if mantissa.unsigned_abs() >= 10u128.pow(P as u32) {
            return Err(decimal_precision_error(mantissa, P, S));
        }
        self.0.push(mantissa);
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.0.push(0);
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let array = primitive_array(self.0, validity)?;
        out.push(
            Int128Chunked::with_chunk(name, array)
                .into_decimal_unchecked(P, S)
                .into_series()
                .into(),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Datetime
// ---------------------------------------------------------------------------

/// A Polars time unit chosen at the type level.
pub trait Unit {
    /// The Polars time unit.
    const UNIT: TimeUnit;
}

/// Milliseconds.
pub struct Ms;
/// Microseconds.
pub struct Us;
/// Nanoseconds.
pub struct Ns;

impl Unit for Ms {
    const UNIT: TimeUnit = TimeUnit::Milliseconds;
}
impl Unit for Us {
    const UNIT: TimeUnit = TimeUnit::Microseconds;
}
impl Unit for Ns {
    const UNIT: TimeUnit = TimeUnit::Nanoseconds;
}

/// Builder for a `Datetime(U, None)` column.
pub struct DatetimeBuilder<U>(Vec<i64>, PhantomData<U>);

impl<U: Unit> DatetimeBuilder<U> {
    fn finish_column(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let array = primitive_array(self.0, validity)?;
        out.push(
            Int64Chunked::with_chunk(name, array)
                .into_datetime(U::UNIT, None)
                .into_series()
                .into(),
        );
        Ok(())
    }
}

#[cfg(feature = "chrono")]
mod chrono_impls {
    use super::{
        Bitmap, Column, ColumnBuilder, DataType, DatetimeBuilder, Fields, Ms, Ns, PhantomData,
        PlSmallStr, PolarsError, PolarsResult, Unit, Us,
    };
    use chrono::{DateTime, TimeZone};

    /// Converts a chrono instant to an integer count of `Self` since the epoch.
    pub trait FromInstant: Unit {
        /// # Errors
        /// Returns an error when the instant is outside the unit's range.
        fn from_instant<Tz: TimeZone>(value: &DateTime<Tz>) -> PolarsResult<i64>;
    }

    impl FromInstant for Ms {
        #[inline(always)]
        fn from_instant<Tz: TimeZone>(value: &DateTime<Tz>) -> PolarsResult<i64> {
            Ok(value.timestamp_millis())
        }
    }

    impl FromInstant for Us {
        #[inline(always)]
        fn from_instant<Tz: TimeZone>(value: &DateTime<Tz>) -> PolarsResult<i64> {
            Ok(value.timestamp_micros())
        }
    }

    #[cold]
    #[inline(never)]
    fn nanos_out_of_range() -> PolarsError {
        PolarsError::ComputeError(
            "df-derive: DateTime<Tz> value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)"
                .into(),
        )
    }

    impl FromInstant for Ns {
        #[inline(always)]
        fn from_instant<Tz: TimeZone>(value: &DateTime<Tz>) -> PolarsResult<i64> {
            value.timestamp_nanos_opt().ok_or_else(nanos_out_of_range)
        }
    }

    impl<Tz: TimeZone, U: FromInstant> ColumnBuilder<DateTime<Tz>> for DatetimeBuilder<U> {
        fn schema(name: PlSmallStr, out: &mut Fields) {
            out.push((name, DataType::Datetime(U::UNIT, None)));
        }

        fn with_capacity(rows: usize) -> Self {
            Self(Vec::with_capacity(rows), PhantomData)
        }

        #[inline(always)]
        fn push(&mut self, value: &DateTime<Tz>) -> PolarsResult<()> {
            self.0.push(U::from_instant(value)?);
            Ok(())
        }

        #[inline(always)]
        fn push_null(&mut self) {
            self.0.push(0);
        }

        fn finish(
            self,
            name: PlSmallStr,
            validity: Option<&Bitmap>,
            out: &mut Vec<Column>,
        ) -> PolarsResult<()> {
            self.finish_column(name, validity, out)
        }
    }
}

#[cfg(feature = "chrono")]
pub use chrono_impls::FromInstant;
