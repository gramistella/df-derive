use std::num::NonZero;

use polars::prelude::{
    BooleanChunked, Column, DataType, IntoSeries, PlSmallStr, PolarsError, PolarsResult, Series,
};
use polars_arrow::array::{BooleanArray, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::types::NativeType;

use super::bits;
use crate::dataframe::{ColumnBuilder, Columnar, Fields};

/// Native numeric types that map one-to-one onto a Polars dtype.
pub trait Primitive: NativeType {
    /// The Polars dtype of a column of this type.
    const DTYPE: DataType;

    /// Wraps a finished array as a named series of [`Primitive::DTYPE`].
    fn series(name: PlSmallStr, array: PrimitiveArray<Self>) -> Series;
}

/// Builds a native array, reporting a length mismatch as an error.
pub(super) fn primitive_array<P: NativeType>(
    values: Vec<P>,
    validity: Option<&Bitmap>,
) -> PolarsResult<PrimitiveArray<P>> {
    PrimitiveArray::try_new(P::PRIMITIVE.into(), values.into(), validity.cloned())
}

/// Appends one converted value per slot, and the default value for a slot
/// that holds none.
///
/// The whole run goes through one `Vec::extend`, which sizes the vector
/// before the loop starts, so an iteration is a conversion and a store. That
/// only works without an early exit: after a failed conversion the remaining
/// slots are filled with the default, and the first error is returned once
/// the loop is done.
#[inline(always)]
pub(super) fn try_extend<'a, T, P, I>(
    out: &mut Vec<P>,
    values: I,
    convert: impl Fn(&'a T) -> PolarsResult<P>,
) -> PolarsResult<()>
where
    T: ?Sized + 'a,
    P: Default,
    I: Iterator<Item = Option<&'a T>>,
{
    let mut failure: Option<PolarsError> = None;
    out.extend(values.map(|value| match value {
        Some(value) if failure.is_none() => convert(value).unwrap_or_else(|error| {
            failure = Some(error);
            P::default()
        }),
        _ => P::default(),
    }));
    failure.map_or(Ok(()), Err)
}

/// The value a slot stores: its own, or the default as a placeholder.
#[inline(always)]
fn stored<P: Copy + Default>(slot: Option<&P>) -> P {
    slot.copied().unwrap_or_default()
}

/// Builder for a column of native numeric values.
pub struct PrimitiveBuilder<P>(Vec<P>);

impl<P: Primitive> PrimitiveBuilder<P> {
    fn finish_column(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        out.push(P::series(name, primitive_array(self.0, validity)?).into());
        Ok(())
    }
}

impl<P: Primitive> ColumnBuilder<P> for PrimitiveBuilder<P> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, P::DTYPE));
    }

    fn new() -> Self {
        Self(Vec::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&P>) -> PolarsResult<()> {
        self.0.push(stored(value));
        Ok(())
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        P: 'a,
        I: Iterator<Item = Option<&'a P>> + Clone,
    {
        self.0.extend(values.map(stored));
        Ok(())
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

macro_rules! primitive {
    ($($ty:ty => $dtype:ident, $chunked:ident);* $(;)?) => {$(
        impl Primitive for $ty {
            const DTYPE: DataType = DataType::$dtype;

            fn series(name: PlSmallStr, array: PrimitiveArray<Self>) -> Series {
                polars::prelude::$chunked::with_chunk(name, array).into_series()
            }
        }

        impl Columnar for $ty {
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

/// A type stored as the native type `$as` after an infallible conversion.
macro_rules! stored_as {
    ($ty:ty => $as:ty, |$value:ident| $convert:expr) => {
        impl ColumnBuilder<$ty> for PrimitiveBuilder<$as> {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                out.push((name, <$as as Primitive>::DTYPE));
            }

            fn new() -> Self {
                Self(Vec::new())
            }

            fn reserve(&mut self, additional: usize) {
                self.0.reserve(additional);
            }

            #[inline(always)]
            fn push(&mut self, value: Option<&$ty>) -> PolarsResult<()> {
                self.0
                    .push(value.map(|$value| $convert).unwrap_or_default());
                Ok(())
            }

            #[inline(always)]
            fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
            where
                I: Iterator<Item = Option<&'a $ty>> + Clone,
            {
                self.0
                    .extend(values.map(|value| value.map(|$value| $convert).unwrap_or_default()));
                Ok(())
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

        impl Columnar for $ty {
            type Builder = PrimitiveBuilder<$as>;
        }
    };
}

macro_rules! non_zero {
    ($($ty:ty),* $(,)?) => {$(
        stored_as!(NonZero<$ty> => $ty, |value| value.get());
    )*};
}

non_zero!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

// Pointer-sized integers are stored as 64-bit columns on every platform so a
// schema does not depend on the build target.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
mod pointer_sized {
    use super::{
        Bitmap, Column, ColumnBuilder, Columnar, Fields, NonZero, PlSmallStr, PolarsResult,
        Primitive, PrimitiveBuilder,
    };

    stored_as!(isize => i64, |value| *value as i64);
    stored_as!(usize => u64, |value| *value as u64);
    stored_as!(NonZero<isize> => i64, |value| value.get() as i64);
    stored_as!(NonZero<usize> => u64, |value| value.get() as u64);
}

/// Builder for a boolean column.
///
/// Values are kept as one byte each while building and packed into a bitmap
/// once at the end.
pub struct BoolBuilder(Vec<bool>);

impl ColumnBuilder<bool> for BoolBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::Boolean));
    }

    fn new() -> Self {
        Self(Vec::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&bool>) -> PolarsResult<()> {
        self.0.push(stored(value));
        Ok(())
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        I: Iterator<Item = Option<&'a bool>> + Clone,
    {
        self.0.extend(values.map(stored));
        Ok(())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let values = bits::pack(&self.0);
        let array = BooleanArray::try_new(ArrowDataType::Boolean, values, validity.cloned())?;
        out.push(BooleanChunked::with_chunk(name, array).into_series().into());
        Ok(())
    }
}

impl Columnar for bool {
    type Builder = BoolBuilder;
}
