//! Shared runtime trait identity for `df-derive`.
//!
//! # What this crate provides
//!
//! `df-derive-core` owns the default `dataframe` traits used by the
//! user-facing `df-derive` facade. Sharing these traits across crates lets
//! derived models compose as nested `ToDataFrame` types without each crate
//! inventing a local runtime identity.
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
//! Most users get this crate through `df-derive`. Depend on this crate
//! directly when you want the shared traits without the facade, or when you
//! use `df-derive-macros` directly and still want the default runtime
//! identity.
//!
//! ```toml
//! [dependencies]
//! df-derive-core = "0.3"
//! df-derive-macros = "0.3"
//! polars = "0.53"
//! ```
//!
//! Default-runtime generated code uses hidden dependency re-exports from this
//! crate, so direct `polars-arrow` dependencies are not required unless you
//! use a custom runtime.
//!
//! ```ignore
//! use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
//! use df_derive_macros::ToDataFrame;
//!
//! #[derive(ToDataFrame)]
//! struct Trade { symbol: String, price: f64, size: u64 }
//! ```
//!
//! # Validating a custom decimal backend
//!
//! The `Decimal128Encode` contract requires round-half-to-even (banker's
//! rounding) on scale-down. The reference `rust_decimal::Decimal` impl in
//! this crate honours that contract and is checked against Polars' decimal
//! string-cast behavior in this crate's integration tests.

// `polars` pulls a wide transitive dependency tree (ahash, foldhash,
// hashbrown, windows-sys variants, …) where multiple resolved versions are
// unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables, and it would fire ~21 times on
// dependencies entirely outside this crate's control. Allow it here so the
// lint surface stays focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

pub mod dataframe {
    use polars::prelude::{AnyValue, DataFrame, DataType, PolarsResult, Series};

    #[cfg(feature = "rust_decimal")]
    const DECIMAL128_MAX_SCALE: u32 = 38;

    /// Hidden dependency re-exports used by generated code for the default
    /// dataframe runtime. This is not part of the public API surface.
    #[doc(hidden)]
    pub mod __private {
        pub use polars;
        pub use polars_arrow;
    }

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

    /// Columnar batch trait implemented by the derive macro.
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

    fn zero_column_dataframe_with_height(n: usize) -> PolarsResult<DataFrame> {
        let dummy = Series::new_empty("_dummy".into(), &DataType::Null)
            .extend_constant(AnyValue::Null, n)?;
        let mut df = DataFrame::new_infer_height(vec![dummy.into()])?;
        df.drop_in_place("_dummy")?;
        Ok(df)
    }

    // Unit-type support for generic payloads such as `Wrapper<()>`. Direct
    // derived fields of type `()` are rejected by df-derive, but a generic
    // field instantiated as `()` contributes zero columns. The
    // `to_dataframe` / `columnar_to_dataframe` paths must still produce a
    // DataFrame with the correct row count, so we use a temporary dummy
    // column that is dropped immediately after construction.
    impl ToDataFrame for () {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            zero_column_dataframe_with_height(1)
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            DataFrame::new_infer_height(vec![])
        }

        fn schema() -> PolarsResult<Vec<(String, DataType)>> {
            Ok(Vec::new())
        }
    }

    impl Columnar for () {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            zero_column_dataframe_with_height(items.len())
        }

        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
            zero_column_dataframe_with_height(items.len())
        }
    }

    /// Plug-in trait for converting a decimal value into its `i128`
    /// mantissa rescaled to a target scale.
    ///
    /// Implementers MUST use round-half-to-even (banker's rounding) on
    /// scale-down so the bytes the derive emits match polars's own
    /// `str_to_dec128` path. A `None` return surfaces as a polars
    /// `ComputeError` from the generated code.
    ///
    /// The codegen invokes this method through UFCS on the selected trait
    /// path, so inherent methods with the same name cannot bypass this
    /// contract. Custom backends (`bigdecimal::BigDecimal`,
    /// arbitrary-precision types, …) provide their own impls; this crate
    /// ships a `rust_decimal::Decimal` impl below.
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
    /// scale-up overflow by this repository's `df-derive-core` integration
    /// tests.
    #[cfg(feature = "rust_decimal")]
    impl Decimal128Encode for rust_decimal::Decimal {
        #[inline]
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
            // Bounds: `rust_decimal::Decimal::scale()` is capped at 28,
            // polars caps decimal scale at `DECIMAL128_MAX_SCALE`, so the
            // scale-up `diff` is at most 38 and the scale-down `diff` is at
            // most 28.
            // `10i128.pow(diff)` therefore fits in i128 for either direction
            // (max `10^38 < 2^127`).
            if target_scale > DECIMAL128_MAX_SCALE {
                return None;
            }

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
