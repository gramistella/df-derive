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
//! - [`dataframe::RowBatch`] — the borrowed-row abstraction accepted by the
//!   encoder.
//! - [`dataframe::Columnar`] — the single batch-encoding primitive populated
//!   by the derive.
//! - [`dataframe::ToDataFrame`] — the per-instance and schema API derived
//!   uniformly from `Columnar`.
//! - [`dataframe::ToDataFrameVec`] — the slice extension trait that routes
//!   `[T]::to_dataframe()` through `Columnar::encode`.
//! - [`dataframe::Decimal128Encode`] — the contract for encoding a decimal
//!   value as an `i128` mantissa rescaled to a target scale. The reference
//!   `rust_decimal::Decimal` impl is gated behind the `rust_decimal`
//!   feature (enabled by default).
//! - `impl Columnar for ()` — the zero-column payload behavior used by
//!   generic `Wrapper<()>` shapes. The blanket implementation supplies
//!   `ToDataFrame`.
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
//! df-derive-core = "0.4"
//! df-derive-macros = "0.4"
//! polars = "0.54"
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
    use polars::prelude::{DataFrame, PolarsResult, SchemaRef};

    #[cfg(feature = "rust_decimal")]
    const DECIMAL128_MAX_SCALE: u32 = 38;

    /// Hidden dependency re-exports used by generated code for the default
    /// dataframe runtime. This is not part of the public API surface.
    #[doc(hidden)]
    pub mod __private {
        pub use polars;
        pub use polars_arrow;
    }

    /// Borrowed rows accepted by [`Columnar::encode`].
    ///
    /// Implementations may project rows from another batch without allocating
    /// an intermediate `Vec<&T>`. The runtime provides the two fundamental
    /// representations: direct rows (`[T]`) and borrowed rows (`[&T]`).
    ///
    /// # Implementation contract
    ///
    /// [`RowBatch::len`] must equal the exact length reported by
    /// [`RowBatch::iter`]. Repeated calls to `iter` must visit the same rows in
    /// the same order. Generated encoders may traverse a batch more than once
    /// and use `len` independently for capacities, offsets, validity, and
    /// zero-column frame heights.
    pub trait RowBatch<T: ?Sized> {
        /// Number of rows in the batch.
        fn len(&self) -> usize;

        /// Whether the batch contains no rows.
        #[inline]
        fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Iterate over every row by reference.
        fn iter<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a T> + 'a
        where
            T: 'a;
    }

    impl<T> RowBatch<T> for [T] {
        #[inline]
        fn len(&self) -> usize {
            <[T]>::len(self)
        }

        #[inline]
        fn iter<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a T> + 'a
        where
            T: 'a,
        {
            <[T]>::iter(self)
        }
    }

    impl<T: ?Sized> RowBatch<T> for [&T] {
        #[inline]
        fn len(&self) -> usize {
            <[&T]>::len(self)
        }

        #[inline]
        fn iter<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a T> + 'a
        where
            T: 'a,
        {
            <[&T]>::iter(self).copied()
        }
    }

    /// Convenience API derived from the single [`Columnar::encode`] runtime
    /// primitive.
    pub trait ToDataFrame: Columnar {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        #[inline]
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            Self::encode(std::slice::from_ref(self))
        }

        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        #[inline]
        fn empty_dataframe() -> PolarsResult<DataFrame> {
            Self::encode(&[] as &[Self])
        }

        /// # Errors
        /// Returns an error if schema generation fails.
        #[inline]
        fn schema() -> PolarsResult<SchemaRef> {
            Ok(Self::empty_dataframe()?.schema().clone())
        }
    }

    impl<T: Columnar> ToDataFrame for T {}

    /// Single columnar batch primitive implemented by the derive macro.
    ///
    /// # Implementation contract
    ///
    /// An implementation must return exactly `rows.len()` rows. Its complete
    /// schema — width, ordered column names, and data types — must be identical
    /// for every batch, including the empty batch used by
    /// [`ToDataFrame::schema`]. It must not return undeclared extra columns or
    /// omit declared columns. Derived parents validate these invariants before
    /// consuming a manually implemented nested encoder.
    pub trait Columnar: Sized {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn encode<B>(rows: &B) -> PolarsResult<DataFrame>
        where
            B: RowBatch<Self> + ?Sized;
    }

    /// Extension trait enabling `.to_dataframe()` on slices (and `Vec` via auto-deref).
    pub trait ToDataFrameVec {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar,
    {
        #[inline]
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            <T as Columnar>::encode(self)
        }
    }

    // Unit-type support for generic payloads such as `Wrapper<()>`. Direct
    // derived fields of type `()` are rejected by df-derive, but a generic
    // field instantiated as `()` contributes zero columns while preserving
    // the batch height.
    impl Columnar for () {
        #[inline]
        fn encode<B>(rows: &B) -> PolarsResult<DataFrame>
        where
            B: RowBatch<Self> + ?Sized,
        {
            Ok(DataFrame::empty_with_height(rows.len()))
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

    impl<T> Decimal128Encode for &T
    where
        T: Decimal128Encode + ?Sized,
    {
        #[inline]
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
            <T as Decimal128Encode>::try_to_i128_mantissa(*self, target_scale)
        }
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
