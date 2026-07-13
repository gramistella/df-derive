// Local runtime fixtures mirror the minimum trait surface documented for
// custom runtimes, so they intentionally omit production API docs.
#![allow(clippy::missing_errors_doc)]

#[allow(dead_code)]
pub mod dataframe {
    use polars::prelude::{DataFrame, PolarsResult, SchemaRef};

    pub trait RowBatch<T: ?Sized> {
        fn len(&self) -> usize;

        fn is_empty(&self) -> bool {
            self.len() == 0
        }

        fn iter<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a T> + 'a
        where
            T: 'a;
    }

    impl<T> RowBatch<T> for [T] {
        fn len(&self) -> usize {
            <[T]>::len(self)
        }

        fn iter<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a T> + 'a
        where
            T: 'a,
        {
            <[T]>::iter(self)
        }
    }

    impl<T: ?Sized> RowBatch<T> for [&T] {
        fn len(&self) -> usize {
            <[&T]>::len(self)
        }

        fn iter<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a T> + 'a
        where
            T: 'a,
        {
            <[&T]>::iter(self).copied()
        }
    }

    pub trait Columnar: Sized {
        fn encode<B>(rows: &B) -> PolarsResult<DataFrame>
        where
            B: RowBatch<Self> + ?Sized;
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
        fn encode<B>(rows: &B) -> PolarsResult<DataFrame>
        where
            B: RowBatch<Self> + ?Sized,
        {
            Ok(DataFrame::empty_with_height(rows.len()))
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
}
