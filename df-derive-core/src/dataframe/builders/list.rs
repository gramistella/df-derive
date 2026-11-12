use polars::prelude::{Column, DataType, PlSmallStr, PolarsError, PolarsResult, Series};
use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::{Offsets, OffsetsBuffer};

use crate::dataframe::frame::{extend_from_slice, first_failure, in_element};
use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Wraps every column of another builder in a list.
///
/// Each sequence appends its elements to the inner builder and records where
/// they end, so a sequence of structs becomes one list column per struct
/// field, all sharing the same offsets. An empty slot is an empty list.
pub struct ListBuilder<B> {
    inner: B,
    offsets: ListOffsets,
}

/// Where each list ends in the buffer its elements share.
///
/// Arrow's own offsets type does the bookkeeping: it only ever appends an
/// end that is at or past the one before, so the finished offsets need no
/// validation.
pub(super) struct ListOffsets {
    ends: Offsets<i64>,
    /// Elements counted by `measure` that `reserve` has not yet made room for.
    measured: usize,
}

impl ListOffsets {
    pub(super) fn new() -> Self {
        Self {
            ends: Offsets::new(),
            measured: 0,
        }
    }

    /// Counts the elements of lists that are about to be recorded.
    #[inline(always)]
    pub(super) fn measure(&mut self, lens: impl Iterator<Item = usize>) {
        self.measured += lens.sum::<usize>();
    }

    /// Makes room for `additional` more lists and returns how many elements
    /// were measured for them, which is the room their owner needs.
    pub(super) fn reserve(&mut self, additional: usize) -> usize {
        self.ends.reserve(additional);
        std::mem::take(&mut self.measured)
    }

    /// Records a list of `len` elements.
    #[inline(always)]
    pub(super) fn push(&mut self, len: usize) -> PolarsResult<()> {
        self.ends.try_push(len)
    }

    /// Records one list per length.
    #[inline(always)]
    pub(super) fn extend(&mut self, lens: impl Iterator<Item = usize>) -> PolarsResult<()> {
        self.ends.try_extend_from_lengths(lens)
    }

    pub(super) fn finish(self) -> OffsetsBuffer<i64> {
        self.ends.into()
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
    let offsets = offsets.finish();
    for column in &mut out[start..] {
        // A list holds its values in one array. The provided builders hand
        // over a single chunk already; a custom one may not.
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

/// The length a slot adds to the element buffer.
#[inline(always)]
fn len<T>(list: Option<&[T]>) -> usize {
    list.map_or(0, <[T]>::len)
}

/// The slot of the slice a `Vec` in a slot holds.
#[inline(always)]
fn as_slice<T>(list: Option<&Vec<T>>) -> Option<&[T]> {
    list.map(Vec::as_slice)
}

/// Names the element a failed list could not append.
#[cold]
#[inline(never)]
fn failed_element<T, B: ColumnBuilder<T>>(list: &[T], error: PolarsError) -> PolarsError {
    match first_failure::<T, T, B>(list) {
        Some((index, error)) => in_element(error, index),
        None => error,
    }
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<[T]> for ListBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        let start = out.len();
        B::schema(name, out);
        wrap_schema(out, start);
    }

    fn new() -> Self {
        Self {
            inner: B::new(),
            offsets: ListOffsets::new(),
        }
    }

    #[inline(always)]
    fn measure<'a, I>(&mut self, values: I)
    where
        [T]: 'a,
        I: Iterator<Item = Option<&'a [T]>> + Clone,
    {
        self.offsets.measure(values.clone().map(len));
        self.inner.measure(values.flatten().flatten().map(Some));
    }

    fn reserve(&mut self, additional: usize) {
        let elements = self.offsets.reserve(additional);
        self.inner.reserve(elements);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&[T]>) -> PolarsResult<()> {
        let list = value.unwrap_or_default();
        if let Err(error) = extend_from_slice(&mut self.inner, list) {
            return Err(failed_element::<T, B>(list, error));
        }
        self.offsets.push(list.len())
    }

    // One list at a time: the elements of a list sit together in memory, and
    // each list fills every inner buffer while it is in cache.
    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        [T]: 'a,
        I: Iterator<Item = Option<&'a [T]>> + Clone,
    {
        for list in values.clone().flatten() {
            extend_from_slice(&mut self.inner, list)?;
        }
        self.offsets.extend(values.map(len))
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let start = out.len();
        self.inner.finish(name, None, out)?;
        wrap_columns(out, start, self.offsets, validity)
    }
}

/// A `Vec` is encoded as the slice it holds.
impl<T, B: ColumnBuilder<T>> ColumnBuilder<Vec<T>> for ListBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <Self as ColumnBuilder<[T]>>::schema(name, out);
    }

    fn new() -> Self {
        <Self as ColumnBuilder<[T]>>::new()
    }

    #[inline(always)]
    fn measure<'a, I>(&mut self, values: I)
    where
        Vec<T>: 'a,
        I: Iterator<Item = Option<&'a Vec<T>>> + Clone,
    {
        <Self as ColumnBuilder<[T]>>::measure(self, values.map(as_slice));
    }

    fn reserve(&mut self, additional: usize) {
        <Self as ColumnBuilder<[T]>>::reserve(self, additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&Vec<T>>) -> PolarsResult<()> {
        <Self as ColumnBuilder<[T]>>::push(self, as_slice(value))
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        Vec<T>: 'a,
        I: Iterator<Item = Option<&'a Vec<T>>> + Clone,
    {
        <Self as ColumnBuilder<[T]>>::extend(self, values.map(as_slice))
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        <Self as ColumnBuilder<[T]>>::finish(self, name, validity, out)
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
