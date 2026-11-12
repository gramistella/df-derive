use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use super::bits;
use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Adds null tracking on top of another builder.
///
/// Every slot is passed on to the inner builder, which appends a placeholder
/// when the slot is empty, and a flag recording whether it held a value is
/// kept here. At `finish` the flags are packed into a bitmap, combined with
/// any validity handed down from an outer option, and applied to every inner
/// column, so nested options collapse to one null per slot.
pub struct OptionBuilder<B> {
    inner: B,
    valid: Vec<bool>,
}

/// A slot of options holds a value only when the slot does and the option in
/// it does.
#[inline(always)]
fn present<T>(slot: Option<&Option<T>>) -> Option<&T> {
    slot.and_then(Option::as_ref)
}

/// The flag kept for a slot: whether it holds a value.
#[inline(always)]
const fn is_valid<T: ?Sized>(slot: Option<&T>) -> bool {
    slot.is_some()
}

impl<T, B: ColumnBuilder<T>> ColumnBuilder<Option<T>> for OptionBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        B::schema(name, out);
    }

    fn new() -> Self {
        Self {
            inner: B::new(),
            valid: Vec::new(),
        }
    }

    #[inline(always)]
    fn measure<'a, I>(&mut self, values: I)
    where
        Option<T>: 'a,
        I: Iterator<Item = Option<&'a Option<T>>> + Clone,
    {
        self.inner.measure(values.map(present));
    }

    fn reserve(&mut self, additional: usize) {
        self.valid.reserve(additional);
        self.inner.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&Option<T>>) -> PolarsResult<()> {
        let value = present(value);
        self.valid.push(is_valid(value));
        self.inner.push(value)
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        Option<T>: 'a,
        I: Iterator<Item = Option<&'a Option<T>>> + Clone,
    {
        let values = values.map(present);
        self.valid.extend(values.clone().map(is_valid));
        self.inner.extend(values)
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        // A column without nulls carries no bitmap at all.
        let own = self.valid.contains(&false).then(|| bits::pack(&self.valid));
        let combined = combine_validities_and(validity, own.as_ref());
        self.inner.finish(name, combined.as_ref(), out)
    }
}

impl<T: Columnar> Columnar for Option<T> {
    type Builder = OptionBuilder<T::Builder>;
}

impl<T: Record> Record for Option<T> {}
