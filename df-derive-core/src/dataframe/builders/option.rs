use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::compute::utils::combine_validities_and;

use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Adds a validity bitmap on top of another builder.
///
/// A `None` pushes a placeholder into the inner builder and a cleared bit
/// here. At `finish` the bitmap is combined with any validity handed down
/// from an outer option and applied to every inner column, so nested options
/// collapse to one null per slot.
pub struct OptionBuilder<B> {
    inner: B,
    validity: BitmapBuilder,
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<Option<T>> for OptionBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        B::schema(name, out);
    }

    fn with_capacity(rows: usize) -> Self {
        Self {
            inner: B::with_capacity(rows),
            validity: BitmapBuilder::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &Option<T>) -> PolarsResult<()> {
        if let Some(value) = value {
            self.inner.push(value)?;
            self.validity.push(true);
        } else {
            self.inner.push_null();
            self.validity.push(false);
        }
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.inner.push_null();
        self.validity.push(false);
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let own = self.validity.into_opt_validity();
        let combined = combine_validities_and(validity, own.as_ref());
        self.inner.finish(name, combined.as_ref(), out)
    }
}

impl<T: Columnar> Columnar for Option<T> {
    type Builder = OptionBuilder<T::Builder>;
}

impl<T: Record> Record for Option<T> {}
