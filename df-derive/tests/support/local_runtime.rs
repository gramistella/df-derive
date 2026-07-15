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

    /// Iterator boundary for generated encoders that may revisit yielded rows.
    ///
    /// # Safety
    ///
    /// `yielded` must equal the number of successful `next` calls. After
    /// `enable_replay`, `replay` must yield those items exactly once in source
    /// order. Generated exact-capacity writes rely on this contract.
    pub unsafe trait RowCursor: Iterator {
        type Replay<'cursor>: Iterator<Item = Self::Item>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, capacity: usize);

        fn replay(&self) -> Self::Replay<'_>;

        fn yielded(&self) -> usize;
    }

    struct StreamingCursor<I>
    where
        I: Iterator,
    {
        inner: I,
        yielded: usize,
    }

    impl<I> StreamingCursor<I>
    where
        I: Iterator,
    {
        const fn new(inner: I) -> Self {
            Self { inner, yielded: 0 }
        }
    }

    impl<I> Iterator for StreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
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

    unsafe impl<I> RowCursor for StreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
    {
        type Replay<'cursor>
            = std::iter::Empty<I::Item>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, _capacity: usize) {
            panic!("df-derive: compact streaming cursor does not support row replay");
        }

        fn replay(&self) -> Self::Replay<'_> {
            std::iter::empty()
        }

        fn yielded(&self) -> usize {
            self.yielded
        }
    }

    struct ReplayStreamingCursor<I>
    where
        I: Iterator,
    {
        inner: I,
        yielded: usize,
        replay: Option<Vec<I::Item>>,
    }

    impl<I> ReplayStreamingCursor<I>
    where
        I: Iterator,
    {
        const fn new(inner: I) -> Self {
            Self {
                inner,
                yielded: 0,
                replay: None,
            }
        }
    }

    impl<I> Iterator for ReplayStreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
    {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.inner.next()?;
            match &mut self.replay {
                Some(replay) => replay.push(item),
                None => self.yielded += 1,
            }
            Some(item)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    unsafe impl<I> RowCursor for ReplayStreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
    {
        type Replay<'cursor>
            = std::iter::Copied<std::slice::Iter<'cursor, I::Item>>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, capacity: usize) {
            assert!(
                self.yielded == 0 && self.replay.is_none(),
                "df-derive: row replay must be enabled before iteration",
            );
            self.replay = Some(Vec::with_capacity(capacity));
        }

        fn replay(&self) -> Self::Replay<'_> {
            self.replay
                .as_ref()
                .expect("df-derive: row replay requested before it was enabled")
                .iter()
                .copied()
        }

        fn yielded(&self) -> usize {
            self.replay.as_ref().map_or(self.yielded, Vec::len)
        }
    }

    struct SliceCursor<'row, T> {
        original: &'row [T],
        remaining: std::slice::Iter<'row, T>,
    }

    impl<'row, T> SliceCursor<'row, T> {
        fn new(rows: &'row [T]) -> Self {
            Self {
                original: rows,
                remaining: rows.iter(),
            }
        }
    }

    impl<'row, T> Iterator for SliceCursor<'row, T> {
        type Item = &'row T;

        fn next(&mut self) -> Option<Self::Item> {
            self.remaining.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.remaining.size_hint()
        }
    }

    unsafe impl<'row, T> RowCursor for SliceCursor<'row, T> {
        type Replay<'cursor>
            = std::slice::Iter<'row, T>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, _capacity: usize) {}

        fn replay(&self) -> Self::Replay<'_> {
            let original: &'row [T] = self.original;
            original[..self.yielded()].iter()
        }

        fn yielded(&self) -> usize {
            self.original.len() - self.remaining.len()
        }
    }

    struct RefSliceCursor<'slice, 'row, T> {
        original: &'slice [&'row T],
        remaining: std::iter::Copied<std::slice::Iter<'slice, &'row T>>,
    }

    impl<'slice, 'row, T> RefSliceCursor<'slice, 'row, T> {
        fn new(rows: &'slice [&'row T]) -> Self {
            Self {
                original: rows,
                remaining: rows.iter().copied(),
            }
        }
    }

    impl<'row, T> Iterator for RefSliceCursor<'_, 'row, T> {
        type Item = &'row T;

        fn next(&mut self) -> Option<Self::Item> {
            self.remaining.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.remaining.size_hint()
        }
    }

    unsafe impl<'slice, 'row, T> RowCursor for RefSliceCursor<'slice, 'row, T> {
        type Replay<'cursor>
            = std::iter::Copied<std::slice::Iter<'slice, &'row T>>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, _capacity: usize) {}

        fn replay(&self) -> Self::Replay<'_> {
            let original: &'slice [&'row T] = self.original;
            original[..self.yielded()].iter().copied()
        }

        fn yielded(&self) -> usize {
            self.original.len() - self.remaining.len()
        }
    }

    pub trait ColumnarSpec: Sized {
        const REQUIRES_ROW_REPLAY: bool = false;

        fn build_schema() -> PolarsResult<SchemaRef>;

        fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
        where
            Self: 'a,
            I: RowCursor<Item = &'a Self>;
    }

    #[allow(
        clippy::inline_always,
        reason = "the cursor type must specialize away before entering generated row loops"
    )]
    #[inline(always)]
    fn encode_streaming_cursor<'row, T, C>(
        mut rows: C,
        sink: &mut ColumnSink,
    ) -> PolarsResult<usize>
    where
        T: ColumnarSpec + 'row,
        C: RowCursor<Item = &'row T>,
    {
        T::encode_columns(&mut rows, sink)?;
        rows.by_ref().for_each(drop);
        Ok(rows.yielded())
    }

    pub trait Columnar: ColumnarSpec {
        fn encode_batch<'a, R>(rows: R) -> PolarsResult<EncodedBatch>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>,
        {
            let schema = <Self as ColumnarSpec>::build_schema()?;
            let mut sink = ColumnSink::new(schema, std::any::type_name::<Self>());
            let rows = rows.into_iter();
            let height = if <Self as ColumnarSpec>::REQUIRES_ROW_REPLAY {
                encode_streaming_cursor::<Self, _>(ReplayStreamingCursor::new(rows), &mut sink)?
            } else {
                encode_streaming_cursor::<Self, _>(StreamingCursor::new(rows), &mut sink)?
            };
            sink.finish(height)
        }

        fn encode_slice(rows: &[Self]) -> PolarsResult<EncodedBatch> {
            let schema = <Self as ColumnarSpec>::build_schema()?;
            let mut sink = ColumnSink::new(schema, std::any::type_name::<Self>());
            let mut rows_cursor = SliceCursor::new(rows);
            <Self as ColumnarSpec>::encode_columns(&mut rows_cursor, &mut sink)?;
            sink.finish(rows.len())
        }

        fn encode_ref_batch<'row>(rows: &[&'row Self]) -> PolarsResult<EncodedBatch>
        where
            Self: 'row,
        {
            let schema = <Self as ColumnarSpec>::build_schema()?;
            let mut sink = ColumnSink::new(schema, std::any::type_name::<Self>());
            let mut rows_cursor = RefSliceCursor::new(rows);
            <Self as ColumnarSpec>::encode_columns(&mut rows_cursor, &mut sink)?;
            rows_cursor.by_ref().for_each(drop);
            sink.finish(rows.len())
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
            Self::encode_slice(std::slice::from_ref(self))?.into_dataframe()
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            Self::encode_slice(&[])?.into_dataframe()
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
            <T as Columnar>::encode_slice(self)?.into_dataframe()
        }
    }

    impl ColumnarSpec for () {
        fn build_schema() -> PolarsResult<SchemaRef> {
            Ok(Arc::new(Schema::default()))
        }

        fn encode_columns<'a, I>(_rows: &mut I, _sink: &mut ColumnSink) -> PolarsResult<()>
        where
            Self: 'a,
            I: RowCursor<Item = &'a Self>,
        {
            Ok(())
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
}
