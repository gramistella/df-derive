use polars::prelude::{Column, DataType, PlSmallStr, PolarsResult, Series};
use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::{Offsets, OffsetsBuffer};

use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Wraps every column of another builder in a list.
///
/// Each pushed sequence appends its elements to the inner builder and records
/// one offset, so a sequence of structs becomes one list column per struct
/// field, all sharing the same offsets.
pub struct ListBuilder<B> {
    inner: B,
    offsets: Offsets<i64>,
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
    offsets: Offsets<i64>,
    validity: Option<&Bitmap>,
) -> PolarsResult<()> {
    let offsets: OffsetsBuffer<i64> = offsets.into();
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

    fn with_capacity(rows: usize) -> Self {
        Self {
            inner: B::with_capacity(rows),
            offsets: Offsets::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &[T]) -> PolarsResult<()> {
        self.inner.extend(value)?;
        self.offsets.try_push(value.len())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.offsets.extend_constant(1);
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

    fn with_capacity(rows: usize) -> Self {
        <Self as ColumnBuilder<[T]>>::with_capacity(rows)
    }

    #[inline(always)]
    fn push(&mut self, value: &Vec<T>) -> PolarsResult<()> {
        <Self as ColumnBuilder<[T]>>::push(self, value.as_slice())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.offsets.extend_constant(1);
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
