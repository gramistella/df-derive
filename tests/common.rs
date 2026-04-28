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
        fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
    }

    /// Internal columnar trait mirrored from the main crate. Implemented by the derive macro.
    pub trait Columnar: Sized {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
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

        fn schema() -> PolarsResult<Vec<(&'static str, DataType)>> {
            Ok(Vec::new())
        }
    }

    impl Columnar for () {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let n = items.len();
            let dummy = Series::new_empty("_dummy".into(), &DataType::Null)
                .extend_constant(AnyValue::Null, n)?;
            let mut df = DataFrame::new_infer_height(vec![dummy.into()])?;
            df.drop_in_place("_dummy")?;
            Ok(df)
        }
    }
}
