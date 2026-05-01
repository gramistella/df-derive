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
        /// Returns one `AnyValue` per inner schema column for `self` —
        /// the per-row slice of `to_dataframe()`. The default impl round-trips
        /// through `to_dataframe()`; the derive overrides it to skip the
        /// one-row `DataFrame` allocation.
        ///
        /// # Errors
        /// Returns an error if value extraction fails.
        fn to_inner_values(&self) -> PolarsResult<Vec<AnyValue<'static>>> {
            let df = self.to_dataframe()?;
            let row = df.get(0).unwrap_or_default();
            Ok(row.into_iter().map(AnyValue::into_static).collect())
        }
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
