//! Canonical runtime traits for [`df-derive`].
//!
//! # What this crate provides
//!
//! `df-derive` is a procedural derive macro that emits code naming a
//! user-supplied trait module. The macro itself ships no traits — every
//! example, test, and downstream user historically had to inline its own
//! copy of `ToDataFrame`, `Columnar`, `ToDataFrameVec`, and
//! `Decimal128Encode`. This crate is the canonical reference module so users
//! without a `paft` runtime can just depend on `df-derive-runtime` and point
//! the derive at it.
//!
//! The [`dataframe`] module exposes:
//!
//! - [`dataframe::ToDataFrame`] — the per-instance API the derive populates.
//! - [`dataframe::Columnar`] — the columnar batch API the derive populates.
//! - [`dataframe::ToDataFrameVec`] — the slice extension trait that routes
//!   `[T]::to_dataframe()` through `Columnar` or `empty_dataframe`.
//! - [`dataframe::Decimal128Encode`] — the contract for encoding a decimal
//!   value as an `i128` mantissa rescaled to a target scale. The reference
//!   `rust_decimal::Decimal` impl is gated behind the `rust_decimal`
//!   feature (enabled by default).
//! - `impl ToDataFrame for ()` and `impl Columnar for ()` — the zero-column
//!   payload behavior used by generic `Wrapper<()>` shapes.
//!
//! # When to use this crate
//!
//! Add this crate when you want the macro's trait surface but do not have a
//! `paft` ecosystem dependency. The macro's default discovery cascade tries
//! the `paft` facade first, then `paft-utils`, then a local
//! `crate::core::dataframe` fallback; this crate is for projects that prefer
//! an explicit, canonical runtime path instead of local trait boilerplate.
//!
//! ```toml
//! [dependencies]
//! df-derive = "0.3"
//! df-derive-runtime = "0.3"
//! ```
//!
//! ```ignore
//! use df_derive::ToDataFrame;
//! use df_derive_runtime::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
//!
//! #[derive(ToDataFrame)]
//! #[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
//! struct Trade { symbol: String, price: f64, size: u64 }
//! ```
//!
//! The `#[df_derive(trait = "...")]` attribute auto-infers the `Columnar`
//! and `Decimal128Encode` paths by replacing the last segment of the trait
//! path, so you only need to name the `ToDataFrame` path.
//!
//! # Validating a custom decimal backend
//!
//! The `Decimal128Encode` contract requires round-half-to-even (banker's
//! rounding) on scale-down. The reference `rust_decimal::Decimal` impl in
//! this crate honours that contract; it is byte-equivalent to the
//! historical `tests/common.rs` impl and is exercised by this repository's
//! unpublished `df-derive-test-harness` workspace crate.
//!
//! [`df-derive`]: https://docs.rs/df-derive

// `polars` pulls a wide transitive dependency tree (ahash, foldhash,
// hashbrown, windows-sys variants, …) where multiple resolved versions are
// unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables, and it would fire ~21 times on
// dependencies entirely outside this crate's control. Allow it here so the
// lint surface stays focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

#[allow(dead_code)]
pub mod dataframe {
    use polars::prelude::{AnyValue, DataFrame, DataType, NamedFrom, PolarsResult, Series};
    pub trait ToDataFrame {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        /// # Errors
        /// Returns an error if schema generation fails.
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
    }

    /// Internal columnar trait mirrored from the main crate. Implemented by the derive macro.
    pub trait Columnar: Sized {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let refs: Vec<&Self> = items.iter().collect();
            Self::columnar_from_refs(&refs)
        }
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
    }

    /// Extension trait enabling `.to_dataframe()` on slices (and `Vec` via auto-deref)
    pub trait ToDataFrameVec {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar + ToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as ToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }

    // Unit-type support: a struct field of type `()` contributes zero columns.
    // The `to_dataframe` / `columnar_to_dataframe` paths must still produce a
    // DataFrame with the correct row count, so we use a temporary dummy column
    // that is dropped immediately after construction.
    impl ToDataFrame for () {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            let dummy = Series::new("_dummy".into(), &[0i32]);
            let mut df = DataFrame::new_infer_height(vec![dummy.into()])?;
            df.drop_in_place("_dummy")?;
            Ok(df)
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            DataFrame::new_infer_height(vec![])
        }

        fn schema() -> PolarsResult<Vec<(String, DataType)>> {
            Ok(Vec::new())
        }
    }

    impl Columnar for () {
        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
            let n = items.len();
            let dummy = Series::new_empty("_dummy".into(), &DataType::Null)
                .extend_constant(AnyValue::Null, n)?;
            let mut df = DataFrame::new_infer_height(vec![dummy.into()])?;
            df.drop_in_place("_dummy")?;
            Ok(df)
        }
    }

    /// Plug-in trait for converting a decimal value into its `i128`
    /// mantissa rescaled to a target scale.
    ///
    /// Implementers MUST use round-half-to-even (banker's rounding) on
    /// scale-down so the bytes the derive emits match polars's own
    /// `str_to_dec128` path. A `None` return surfaces as a polars
    /// `ComputeError` from the generated code, matching the historical
    /// scale-up overflow path.
    ///
    /// The codegen invokes the method via dot syntax (after an anonymous
    /// `use … as _;` import), so method resolution selects the impl from
    /// the value reference's type. Custom backends (`bigdecimal::BigDecimal`,
    /// arbitrary-precision types, …) provide their own impls; this crate
    /// ships a `rust_decimal::Decimal` impl below for tests and benches.
    pub trait Decimal128Encode {
        /// Returns the mantissa as `i128` after rescaling `self` to
        /// `target_scale`, or `None` if the conversion would overflow or
        /// otherwise violate the schema. Implementations MUST round
        /// half-to-even on scale-down.
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }

    /// Reference [`Decimal128Encode`] impl for [`rust_decimal::Decimal`].
    ///
    /// Banker's-rounding contract: round-half-to-even on scale-down,
    /// `checked_mul` overflow-to-`None` on scale-up. This impl is verified
    /// against polars's `str_to_dec128` on a battery of inputs covering
    /// half-tie boundaries (positive and negative), large magnitudes, and
    /// scale-up overflow by this repository's unpublished
    /// `df-derive-test-harness` workspace crate.
    #[cfg(feature = "rust_decimal")]
    impl Decimal128Encode for rust_decimal::Decimal {
        #[inline]
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
            // Bounds: `rust_decimal::Decimal::scale()` is capped at 28,
            // polars caps decimal scale at 38, so the scale-up `diff` is at
            // most 38 and the scale-down `diff` is at most 28.
            // `10i128.pow(diff)` therefore fits in i128 for either direction
            // (max `10^38 < 2^127`).
            let source_scale = self.scale();
            let mantissa: i128 = self.mantissa();
            if source_scale == target_scale {
                return Some(mantissa);
            }
            if source_scale < target_scale {
                let diff = target_scale - source_scale;
                let pow = 10i128.pow(diff);
                return mantissa.checked_mul(pow);
            }
            // Scale-down with round-half-to-even on the unsigned magnitude,
            // then re-apply sign — matches polars's `div_128_pow10`
            // semantics. The `(abs / pow)` quotient cannot exceed `i128::MAX`
            // because `abs <= i128::MAX as u128` and `pow >= 1`, so the
            // `cast_signed` is value-preserving.
            let diff = source_scale - target_scale;
            let pow = 10i128.pow(diff).cast_unsigned();
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
}
