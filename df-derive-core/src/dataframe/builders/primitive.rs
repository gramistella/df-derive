use std::num::NonZero;

use polars::prelude::{
    BooleanChunked, Column, DataType, IntoSeries, PlSmallStr, PolarsResult, Series,
};
use polars_arrow::array::{BooleanArray, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::types::NativeType;

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

/// A type stored as the native type `$as` after an infallible conversion.
macro_rules! converted {
    ($ty:ty => $as:ty, |$value:ident| $convert:expr) => {
        impl ColumnBuilder<$ty> for PrimitiveBuilder<$as> {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                out.push((name, <$as as Primitive>::DTYPE));
            }

            fn with_capacity(rows: usize) -> Self {
                Self(Vec::with_capacity(rows))
            }

            #[inline(always)]
            fn push(&mut self, $value: &$ty) -> PolarsResult<()> {
                self.0.push($convert);
                Ok(())
            }

            #[inline(always)]
            fn push_null(&mut self) {
                self.0.push(<$as>::default());
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

macro_rules! non_zero {
    ($($ty:ty),* $(,)?) => {$(
        converted!(NonZero<$ty> => $ty, |value| value.get());
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

    converted!(isize => i64, |value| *value as i64);
    converted!(usize => u64, |value| *value as u64);
    converted!(NonZero<isize> => i64, |value| value.get() as i64);
    converted!(NonZero<usize> => u64, |value| value.get() as u64);
}

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

impl Columnar for bool {
    type Builder = BoolBuilder;
}
