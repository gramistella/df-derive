// Local runtime fixtures mirror the minimum trait surface documented for
// custom runtimes, so they intentionally omit production API docs.
#![allow(clippy::missing_errors_doc)]

#[allow(dead_code)]
pub mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
    }

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let refs: Vec<&Self> = items.iter().collect();
            Self::columnar_from_refs(&refs)
        }

        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
    }

    pub trait ToDataFrameVec {
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

    impl ToDataFrame for () {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            Ok(DataFrame::empty_with_height(1))
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            Ok(DataFrame::empty_with_height(0))
        }

        fn schema() -> PolarsResult<Vec<(String, DataType)>> {
            Ok(Vec::new())
        }
    }

    impl Columnar for () {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            Ok(DataFrame::empty_with_height(items.len()))
        }

        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
            Ok(DataFrame::empty_with_height(items.len()))
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
}
