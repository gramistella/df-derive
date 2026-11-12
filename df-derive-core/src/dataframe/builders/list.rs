use polars::prelude::{Column, DataType, PlSmallStr, PolarsResult, Series};
use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::OffsetsBuffer;

use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Wraps every column of another builder in a list.
///
/// Each pushed sequence appends its elements to the inner builder and records
/// one offset, so a sequence of structs becomes one list column per struct
/// field, all sharing the same offsets.
pub struct ListBuilder<B> {
    inner: B,
    offsets: ListOffsets,
}

/// End offsets of the lists pushed so far.
///
/// The running total lives in the builder instead of being read back from the
/// last stored offset, which would chain every push to the one before it.
pub(super) struct ListOffsets {
    ends: Vec<i64>,
    total: i64,
}

impl ListOffsets {
    pub(super) fn with_capacity(rows: usize) -> Self {
        let mut ends = Vec::with_capacity(rows.saturating_add(1));
        ends.push(0);
        Self { ends, total: 0 }
    }

    /// Records a list of `len` elements.
    #[inline(always)]
    #[allow(clippy::cast_possible_wrap)]
    pub(super) fn push(&mut self, len: usize) {
        // A total past `i64::MAX` wraps negative and is rejected as a
        // non-monotonic offset in `finish`.
        self.total = self.total.wrapping_add(len as i64);
        self.ends.push(self.total);
    }

    /// Records one list per length.
    #[inline(always)]
    #[allow(clippy::cast_possible_wrap)]
    pub(super) fn extend(&mut self, lens: impl Iterator<Item = usize>) {
        let mut total = self.total;
        self.ends.extend(lens.map(|len| {
            total = total.wrapping_add(len as i64);
            total
        }));
        self.total = total;
    }

    /// Records an empty list, the placeholder for a null.
    #[inline(always)]
    pub(super) fn push_empty(&mut self) {
        self.ends.push(self.total);
    }

    pub(super) fn finish(self) -> PolarsResult<OffsetsBuffer<i64>> {
        OffsetsBuffer::try_from(self.ends)
    }
}

/// Wraps the dtypes of `out[start..]` in `List`.
pub(super) fn wrap_schema(out: &mut Fields, start: usize) {
    for (_, dtype) in &mut out[start..] {
        let inner = std::mem::replace(dtype, DataType::Null);
        *dtype = DataType::List(Box::new(inner));
    }
}

/// Wraps the columns `out[start..]` in list columns that share `offsets`.
pub(super) fn wrap_columns(
    out: &mut [Column],
    start: usize,
    offsets: ListOffsets,
    validity: Option<&Bitmap>,
) -> PolarsResult<()> {
    let offsets = offsets.finish()?;
    for column in &mut out[start..] {
        let series = column.as_materialized_series().rechunk();
        let values = series.chunks()[0].clone();
        let list = ListArray::<i64>::try_new(
            ListArray::<i64>::default_datatype(values.dtype().clone()),
            offsets.clone(),
            values,
            validity.cloned(),
        )?;
        let dtype = DataType::List(Box::new(series.dtype().clone()));
        *column =
            Series::from_chunk_and_dtype(series.name().clone(), Box::new(list), &dtype)?.into();
    }
    Ok(())
}

impl<B> ListBuilder<B> {
    fn list_schema<T>(name: PlSmallStr, out: &mut Fields)
    where
        B: ColumnBuilder<T>,
    {
        let start = out.len();
        B::schema(name, out);
        wrap_schema(out, start);
    }

    fn finish_lists<T>(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()>
    where
        B: ColumnBuilder<T>,
    {
        let start = out.len();
        self.inner.finish(name, None, out)?;
        wrap_columns(out, start, self.offsets, validity)
    }
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<[T]> for ListBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        Self::list_schema::<T>(name, out);
    }

    // One counter for this list's total length, then the inner builder's.
    const LISTS: usize = 1 + B::LISTS;
    const STREAMS: usize = 1 + B::STREAMS;

    fn with_capacity(rows: usize) -> Self {
        Self {
            inner: B::with_capacity(rows),
            offsets: ListOffsets::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn measure(value: &[T], counts: &mut [usize]) {
        counts[0] += value.len();
        if B::LISTS > 0 {
            for element in value {
                B::measure(element, &mut counts[1..]);
            }
        }
    }

    fn with_capacities(rows: usize, counts: &[usize]) -> Self {
        Self {
            inner: B::with_capacities(counts[0], &counts[1..]),
            offsets: ListOffsets::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &[T]) -> PolarsResult<()> {
        self.inner.extend(value)?;
        self.offsets.push(value.len());
        Ok(())
    }

    #[inline(always)]
    fn extend_from<'a, I>(&mut self, values: I, _narrow: bool) -> PolarsResult<()>
    where
        [T]: 'a,
        I: Iterator<Item = &'a [T]> + Clone,
    {
        // The inner builder is moved into a local for the loop. As a field
        // of `self` its buffer lengths are re-read from memory on every
        // append; as a local the compiler can keep them in registers.
        let mut inner = std::mem::replace(&mut self.inner, B::with_capacity(0));
        let mut result = Ok(());
        for value in values.clone() {
            if let Err(error) = inner.extend(value) {
                result = Err(error);
                break;
            }
        }
        self.inner = inner;
        result?;
        self.offsets.extend(values.map(<[T]>::len));
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.offsets.push_empty();
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.finish_lists::<T>(name, validity, out)
    }
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<Vec<T>> for ListBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        Self::list_schema::<T>(name, out);
    }

    const LISTS: usize = <Self as ColumnBuilder<[T]>>::LISTS;
    const STREAMS: usize = <Self as ColumnBuilder<[T]>>::STREAMS;

    fn with_capacity(rows: usize) -> Self {
        <Self as ColumnBuilder<[T]>>::with_capacity(rows)
    }

    #[inline(always)]
    fn measure(value: &Vec<T>, counts: &mut [usize]) {
        <Self as ColumnBuilder<[T]>>::measure(value.as_slice(), counts);
    }

    fn with_capacities(rows: usize, counts: &[usize]) -> Self {
        <Self as ColumnBuilder<[T]>>::with_capacities(rows, counts)
    }

    #[inline(always)]
    fn push(&mut self, value: &Vec<T>) -> PolarsResult<()> {
        <Self as ColumnBuilder<[T]>>::push(self, value.as_slice())
    }

    #[inline(always)]
    fn extend_from<'a, I>(&mut self, values: I, narrow: bool) -> PolarsResult<()>
    where
        Vec<T>: 'a,
        I: Iterator<Item = &'a Vec<T>> + Clone,
    {
        <Self as ColumnBuilder<[T]>>::extend_from(self, values.map(Vec::as_slice), narrow)
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.offsets.push_empty();
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.finish_lists::<T>(name, validity, out)
    }
}

impl<T: Columnar> Columnar for Vec<T> {
    type Builder = ListBuilder<T::Builder>;
}

impl<T: Columnar> Columnar for [T] {
    type Builder = ListBuilder<T::Builder>;
}

impl<T: Record> Record for Vec<T> {}
impl<T: Record> Record for [T] {}
