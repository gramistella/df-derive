/// Converts a decimal value into its `i128` mantissa at a target scale.
///
/// Implement this for a decimal type to store it in a Polars `Decimal`
/// column. Implementations MUST round half-to-even when scaling down so the
/// stored bytes match Polars' own string-to-decimal conversion. Returning
/// `None` surfaces as a compute error naming the row and field.
///
/// The trait is about arithmetic only. Whether a mantissa fits the precision
/// of the column it is headed for is checked by
/// [`DecimalBuilder`](super::builders::DecimalBuilder), which knows that
/// precision.
///
/// The trait is part of the `decimal` feature, which `rust_decimal` (on by
/// default) turns on.
pub trait Decimal128Encode {
    /// Returns the mantissa after rescaling `self` to `target_scale`, or
    /// `None` when the value has no `i128` mantissa at that scale.
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
}

impl<T> Decimal128Encode for &T
where
    T: Decimal128Encode + ?Sized,
{
    #[inline]
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        <T as Decimal128Encode>::try_to_i128_mantissa(*self, target_scale)
    }
}

/// Every power of ten an `i128` can hold: `POWERS_OF_TEN[n]` is `10^n`.
///
/// Rescaling multiplies or divides by one of these for every value, so they
/// are looked up rather than computed. The table also bounds the scale: a
/// decimal with more than 38 decimals has no place in it.
#[cfg(feature = "rust_decimal")]
const POWERS_OF_TEN: [i128; 39] = {
    let mut powers = [1; 39];
    let mut exponent = 1;
    while exponent < powers.len() {
        powers[exponent] = powers[exponent - 1] * 10;
        exponent += 1;
    }
    powers
};

/// Reference impl for [`rust_decimal::Decimal`]: half-to-even on scale-down,
/// `None` on scale-up overflow and for a scale above 38. Checked against
/// Polars' decimal string cast in this crate's tests.
#[cfg(feature = "rust_decimal")]
impl Decimal128Encode for rust_decimal::Decimal {
    // Always inlined: a column of decimals is then one loop. Left to the
    // compiler's judgement, a few more lines here turn every value into a
    // call, which was measured at a third more time for such a column.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        if target_scale as usize >= POWERS_OF_TEN.len() {
            return None;
        }

        let source_scale = self.scale();
        let mantissa: i128 = self.mantissa();
        // `rust_decimal` caps its scale at 28 and the target is at most 38,
        // so the difference always has a power in the table.
        let power = *POWERS_OF_TEN.get(source_scale.abs_diff(target_scale) as usize)?;
        if source_scale <= target_scale {
            return mantissa.checked_mul(power);
        }

        // Scale down on the unsigned magnitude, round half-to-even, then
        // restore the sign.
        let power = power.cast_unsigned();
        let magnitude = mantissa.unsigned_abs();
        let quotient = (magnitude / power).cast_signed();
        let remainder = magnitude % power;
        let rounded = match remainder.cmp(&(power / 2)) {
            ::std::cmp::Ordering::Greater => quotient + 1,
            ::std::cmp::Ordering::Less => quotient,
            ::std::cmp::Ordering::Equal => quotient + (quotient & 1),
        };
        Some(if mantissa < 0 { -rounded } else { rounded })
    }
}
