// Local runtime fixtures mirror the minimum trait surface documented for
// custom runtimes, so they intentionally omit production API docs.
#![allow(clippy::missing_errors_doc)]

#[allow(dead_code)]
pub mod dataframe {
    use polars::prelude::{DataFrame, PolarsResult, SchemaRef};

    pub trait Columnar: Sized {
        fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>;
    }

    pub trait ToDataFrame: Columnar {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            Self::encode(std::slice::from_ref(self))
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            Self::encode(&[] as &[Self])
        }

        fn schema() -> PolarsResult<SchemaRef> {
            Ok(Self::empty_dataframe()?.schema().clone())
        }
    }

    impl<T: Columnar> ToDataFrame for T {}

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            <T as Columnar>::encode(self)
        }
    }

    impl Columnar for () {
        fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>,
        {
            Ok(DataFrame::empty_with_height(rows.into_iter().count()))
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
}
