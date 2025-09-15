#[allow(dead_code)]
pub mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};
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
}
