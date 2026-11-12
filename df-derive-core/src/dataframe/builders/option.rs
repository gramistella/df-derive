use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Adds null tracking on top of another builder.
///
/// A `None` pushes a placeholder into the inner builder and `false` here. At
/// `finish` the flags are packed into a bitmap, combined with any validity
/// handed down from an outer option, and applied to every inner column, so
/// nested options collapse to one null per slot.
///
/// Validity is kept as one byte per slot while building and packed once at
/// the end. Appending a byte is a plain vector push, and a whole slice of
/// options can be appended with one tight loop.
pub struct OptionBuilder<B> {
    inner: B,
    valid: Vec<bool>,
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<Option<T>> for OptionBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        B::schema(name, out);
    }

    const LISTS: usize = B::LISTS;

    fn with_capacity(rows: usize) -> Self {
        Self {
            inner: B::with_capacity(rows),
            valid: Vec::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn measure(value: &Option<T>, counts: &mut [usize]) {
        if let Some(value) = value {
            B::measure(value, counts);
        }
    }

    fn with_capacities(rows: usize, counts: &[usize]) -> Self {
        Self {
            inner: B::with_capacities(rows, counts),
            valid: Vec::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &Option<T>) -> PolarsResult<()> {
        if let Some(value) = value {
            self.inner.push(value)?;
            self.valid.push(true);
        } else {
            self.inner.push_null();
            self.valid.push(false);
        }
        Ok(())
    }

    #[inline(always)]
    fn extend_from<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        Option<T>: 'a,
        I: Iterator<Item = &'a Option<T>> + Clone,
    {
        self.valid.extend(values.clone().map(Option::is_some));
        self.inner.extend_nullable_from(values)
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.inner.push_null();
        self.valid.push(false);
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let own = self
            .valid
            .contains(&false)
            .then(|| Bitmap::from(self.valid));
        let combined = combine_validities_and(validity, own.as_ref());
        self.inner.finish(name, combined.as_ref(), out)
    }
}

impl<T: Columnar> Columnar for Option<T> {
    type Builder = OptionBuilder<T::Builder>;
}

impl<T: Record> Record for Option<T> {}
