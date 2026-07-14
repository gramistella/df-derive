// Local runtime fixtures mirror the minimum trait surface documented for
// custom runtimes, so they intentionally omit production API docs.
#![allow(clippy::missing_errors_doc)]

#[allow(dead_code)]
pub mod dataframe {
    use std::sync::Arc;

    use polars::prelude::{Column, DataFrame, PolarsResult, Schema, SchemaRef, polars_err};

    pub struct EncodedBatch {
        height: usize,
        columns: Vec<Column>,
    }

    impl EncodedBatch {
        #[must_use]
        pub fn into_columns(self) -> Vec<Column> {
            self.columns
        }

        fn into_dataframe(self) -> PolarsResult<DataFrame> {
            DataFrame::new(self.height, self.columns)
        }
    }

    pub struct ColumnSink {
        schema: SchemaRef,
        columns: Vec<Column>,
        producer: &'static str,
    }

    impl ColumnSink {
        fn new(schema: SchemaRef, producer: &'static str) -> Self {
            let columns = Vec::with_capacity(schema.len());
            Self {
                schema,
                columns,
                producer,
            }
        }

        pub fn push(&mut self, column: Column) -> PolarsResult<()> {
            let index = self.columns.len();
            let Some((expected_name, expected_dtype)) = self.schema.get_at_index(index) else {
                return Err(polars_err!(
                    ComputeError:
                    "fixture ColumnarSpec for {} exceeded schema width {}",
                    self.producer,
                    self.schema.len(),
                ));
            };
            if column.name() != expected_name {
                return Err(polars_err!(
                    ComputeError:
                    "fixture ColumnarSpec for {} returned column `{}`, expected `{}`",
                    self.producer,
                    column.name(),
                    expected_name,
                ));
            }
            if column.dtype() != expected_dtype {
                return Err(polars_err!(
                    ComputeError:
                    "fixture ColumnarSpec for {} returned dtype {:?}, expected {:?}",
                    self.producer,
                    column.dtype(),
                    expected_dtype,
                ));
            }
            self.columns.push(column);
            Ok(())
        }

        fn finish(self, height: usize) -> PolarsResult<EncodedBatch> {
            if self.columns.len() != self.schema.len() {
                return Err(polars_err!(
                    ComputeError:
                    "fixture ColumnarSpec for {} returned schema width {}, expected {}",
                    self.producer,
                    self.columns.len(),
                    self.schema.len(),
                ));
            }
            if let Some(column) = self.columns.iter().find(|column| column.len() != height) {
                return Err(polars_err!(
                    ComputeError:
                    "fixture ColumnarSpec for {} returned height {}, expected {}",
                    self.producer,
                    column.len(),
                    height,
                ));
            }
            Ok(EncodedBatch {
                height,
                columns: self.columns,
            })
        }
    }

    struct CountingIterator<I> {
        inner: I,
        yielded: usize,
    }

    impl<I> Iterator for CountingIterator<I>
    where
        I: Iterator,
    {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.inner.next()?;
            self.yielded += 1;
            Some(item)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    pub trait ColumnarSpec: Sized {
        fn build_schema() -> PolarsResult<SchemaRef>;

        fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
        where
            Self: 'a,
            I: Iterator<Item = &'a Self>;
    }

    pub trait Columnar: ColumnarSpec {
        fn encode_batch<'a, R>(rows: R) -> PolarsResult<EncodedBatch>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>,
        {
            let schema = <Self as ColumnarSpec>::build_schema()?;
            let mut sink = ColumnSink::new(schema, std::any::type_name::<Self>());
            let mut rows = CountingIterator {
                inner: rows.into_iter(),
                yielded: 0,
            };
            <Self as ColumnarSpec>::encode_columns(&mut rows, &mut sink)?;
            rows.by_ref().for_each(drop);
            sink.finish(rows.yielded)
        }

        fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>,
        {
            Self::encode_batch(rows)?.into_dataframe()
        }
    }

    impl<T: ColumnarSpec> Columnar for T {}

    pub trait ToDataFrame: Columnar {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            Self::encode(std::slice::from_ref(self))
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            Self::encode(&[] as &[Self])
        }

        fn schema() -> PolarsResult<SchemaRef> {
            <Self as ColumnarSpec>::build_schema()
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

    impl ColumnarSpec for () {
        fn build_schema() -> PolarsResult<SchemaRef> {
            Ok(Arc::new(Schema::default()))
        }

        fn encode_columns<'a, I>(_rows: &mut I, _sink: &mut ColumnSink) -> PolarsResult<()>
        where
            Self: 'a,
            I: Iterator<Item = &'a Self>,
        {
            Ok(())
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
}
