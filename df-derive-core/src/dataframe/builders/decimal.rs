use polars::prelude::{
    Column, DataType, Int128Chunked, IntoSeries, PlSmallStr, PolarsError, PolarsResult,
};
use polars_arrow::bitmap::Bitmap;

use super::primitive::primitive_array;
use crate::dataframe::{ColumnBuilder, Decimal128Encode, Fields};

/// Builder for a `Decimal(P, S)` column from any [`Decimal128Encode`] value.
///
/// This is the encoding `#[df_derive(decimal(precision = P, scale = S))]`
/// selects.
pub struct DecimalBuilder<const P: usize, const S: usize>(Vec<i128>);

#[cold]
#[inline(never)]
fn rescale_error(scale: usize) -> PolarsError {
    PolarsError::ComputeError(
        format!(
            "df-derive: decimal mantissa rescale to scale {scale} failed (overflow or precision loss)"
        )
        .into(),
    )
}

#[cold]
#[inline(never)]
fn precision_error(mantissa: i128, precision: usize, scale: usize) -> PolarsError {
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
            return Err(rescale_error(S));
        };
        if mantissa.unsigned_abs() >= 10u128.pow(P as u32) {
            return Err(precision_error(mantissa, P, S));
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

#[cfg(feature = "rust_decimal")]
impl crate::dataframe::Columnar for rust_decimal::Decimal {
    type Builder = DecimalBuilder<38, 10>;
}
