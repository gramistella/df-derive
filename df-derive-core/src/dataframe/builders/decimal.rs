use polars::prelude::{
    Column, DataType, Int128Chunked, IntoSeries, PlSmallStr, PolarsError, PolarsResult,
};
use polars_arrow::bitmap::Bitmap;

use super::primitive::{primitive_array, try_extend};
use crate::dataframe::{ColumnBuilder, Decimal128Encode, Fields};

/// Builder for a `Decimal(P, S)` column from any [`Decimal128Encode`] value.
///
/// This is the encoding `#[df_derive(decimal(precision = P, scale = S))]`
/// selects. A value whose mantissa at scale `S` has more than `P` digits is
/// an error. A `P` or `S` that Polars has no decimal type for does not
/// compile.
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

impl<const P: usize, const S: usize> DecimalBuilder<P, S> {
    /// The value's mantissa at scale `S`, if it fits in `P` digits.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn mantissa<T: Decimal128Encode + ?Sized>(value: &T) -> PolarsResult<i128> {
        let Some(mantissa) = value.try_to_i128_mantissa(S as u32) else {
            return Err(rescale_error(S));
        };
        if mantissa.unsigned_abs() >= 10u128.pow(P as u32) {
            return Err(precision_error(mantissa, P, S));
        }
        Ok(mantissa)
    }
}

impl<T, const P: usize, const S: usize> ColumnBuilder<T> for DecimalBuilder<P, S>
where
    T: Decimal128Encode + ?Sized,
{
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::Decimal(P, S)));
    }

    fn new() -> Self {
        const {
            assert!(
                0 < P && P <= 38 && S <= P,
                "`Decimal(P, S)` needs 1 <= P <= 38 and S <= P"
            );
        }
        Self(Vec::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&T>) -> PolarsResult<()> {
        self.0.push(match value {
            Some(value) => Self::mantissa(value)?,
            None => 0,
        });
        Ok(())
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        T: 'a,
        I: Iterator<Item = Option<&'a T>> + Clone,
    {
        try_extend(&mut self.0, values, Self::mantissa)
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
