#[cfg(feature = "rust_decimal")]
const DECIMAL128_MAX_SCALE: u32 = 38;

/// Converts a decimal value into its `i128` mantissa at a target scale.
///
/// Implement this for a decimal type to store it in a Polars `Decimal`
/// column. Implementations MUST round half-to-even when scaling down so the
/// stored bytes match Polars' own string-to-decimal conversion. Returning
/// `None` surfaces as a compute error naming the row and field.
pub trait Decimal128Encode {
    /// Returns the mantissa after rescaling `self` to `target_scale`, or
    /// `None` if the conversion would overflow.
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

/// Reference impl for [`rust_decimal::Decimal`]: half-to-even on scale-down,
/// `None` on scale-up overflow. Checked against Polars' decimal string cast
/// in this crate's tests.
#[cfg(feature = "rust_decimal")]
impl Decimal128Encode for rust_decimal::Decimal {
    #[inline]
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        // `rust_decimal` caps scale at 28 and Polars caps it at 38, so the
        // power of ten below always fits in `i128`.
        if target_scale > DECIMAL128_MAX_SCALE {
            return None;
        }

        let source_scale = self.scale();
        let mantissa: i128 = self.mantissa();
        if source_scale == target_scale {
            return Some(mantissa);
        }
        if source_scale < target_scale {
            let pow = 10i128.pow(target_scale - source_scale);
            return mantissa.checked_mul(pow);
        }
        // Scale down on the unsigned magnitude, round half-to-even, then
        // restore the sign.
        let pow = 10i128.pow(source_scale - target_scale).cast_unsigned();
        let neg = mantissa < 0;
        let abs = mantissa.unsigned_abs();
        let q = (abs / pow).cast_signed();
        let r = abs % pow;
        let half = pow / 2;
        let rounded = match r.cmp(&half) {
            ::std::cmp::Ordering::Greater => q + 1,
            ::std::cmp::Ordering::Less => q,
            ::std::cmp::Ordering::Equal => q + (q & 1),
        };
        Some(if neg { -rounded } else { rounded })
    }
}
