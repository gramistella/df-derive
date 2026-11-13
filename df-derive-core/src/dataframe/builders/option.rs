use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;

use super::bits;
use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Adds null tracking on top of another builder.
///
/// Every slot is passed on to the inner builder, which appends a placeholder
/// when the slot is empty, and whether the slot held a value is recorded
/// here. At `finish` that record becomes a bitmap, is combined with any
/// validity handed down from an outer option, and applied to every inner
/// column, so nested options collapse to one null per slot.
///
/// How the record is kept depends on how the inner builder is fed. Slots that
/// arrive in runs get a flag each, written by a loop of its own over the run
/// and packed into bits at the end. Slots that arrive one at a time (see
/// [`ROW_AT_A_TIME`](ColumnBuilder::ROW_AT_A_TIME)) are counted, and only the
/// empty ones are recorded.
pub struct OptionBuilder<B> {
    inner: B,
    /// One flag per slot, when the inner builder is fed runs.
    valid: Vec<bool>,
    /// The empty slots, when the inner builder is fed a row at a time.
    gaps: Gaps,
}

/// Which slots are empty, among slots that arrive one at a time.
///
/// A slot that holds a value costs a count and nothing else: the bitmap
/// starts out all set, and an empty slot clears its bit.
struct Gaps {
    /// A bit for every slot there is room for so far, clear where a slot is
    /// empty. Slots past its end hold values.
    filled: MutableBitmap,
    /// The number of slots so far.
    slots: usize,
}

impl Gaps {
    fn new() -> Self {
        Self {
            filled: MutableBitmap::new(),
            slots: 0,
        }
    }

    /// Makes room for `additional` more slots.
    fn reserve(&mut self, additional: usize) {
        let room = (self.slots + additional).saturating_sub(self.filled.len());
        self.filled.extend_constant(room, true);
    }

    /// Counts one more slot.
    #[inline(always)]
    fn push(&mut self, holds_value: bool) {
        if !holds_value {
            if self.slots >= self.filled.len() {
                self.grow();
            }
            self.filled.set(self.slots, false);
        }
        self.slots += 1;
    }

    /// Makes room for the slot being pushed and for as many slots again,
    /// when none was reserved for it.
    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let room = (self.slots + 1).max(2 * self.filled.len()) - self.filled.len();
        self.filled.extend_constant(room, true);
    }

    /// The validity of the slots, or `None` when every slot holds a value.
    fn finish(mut self) -> Option<Bitmap> {
        // The slots after the last one room was made for hold values, and
        // room that was made for slots that never came is cut off.
        let missing = self.slots.saturating_sub(self.filled.len());
        self.filled.extend_constant(missing, true);
        let validity = Bitmap::from(self.filled).sliced(0, self.slots);
        (validity.unset_bits() > 0).then_some(validity)
    }
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
    const ROW_AT_A_TIME: bool = B::ROW_AT_A_TIME;

    fn schema(name: PlSmallStr, out: &mut Fields) {
        B::schema(name, out);
    }

    fn new() -> Self {
        Self {
            inner: B::new(),
            valid: Vec::new(),
            gaps: Gaps::new(),
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
        if B::ROW_AT_A_TIME {
            self.gaps.reserve(additional);
        } else {
            self.valid.reserve(additional);
        }
        self.inner.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&Option<T>>) -> PolarsResult<()> {
        let value = present(value);
        if B::ROW_AT_A_TIME {
            self.gaps.push(is_valid(value));
        } else {
            self.valid.push(is_valid(value));
        }
        self.inner.push(value)
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        Option<T>: 'a,
        I: Iterator<Item = Option<&'a Option<T>>> + Clone,
    {
        let values = values.map(present);
        if B::ROW_AT_A_TIME {
            // A run reaches a builder that is fed by row when it is the items
            // of a list. Its slots are recorded like the column's others.
            for value in values {
                self.gaps.push(is_valid(value));
                self.inner.push(value)?;
            }
            return Ok(());
        }
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
        let own = if B::ROW_AT_A_TIME {
            self.gaps.finish()
        } else {
            self.valid.contains(&false).then(|| bits::pack(&self.valid))
        };
        let combined = combine_validities_and(validity, own.as_ref());
        self.inner.finish(name, combined.as_ref(), out)
    }
}

impl<T: Columnar> Columnar for Option<T> {
    type Builder = OptionBuilder<T::Builder>;
}

impl<T: Record> Record for Option<T> {}

#[cfg(test)]
mod tests {
    use super::Gaps;

    /// Slots pushed one at a time, with room reserved for `reserved` of them
    /// first, come out as the bitmap of which ones held a value.
    fn check(holds_value: &[bool], reserved: &[usize]) {
        let mut gaps = Gaps::new();
        let mut reservations = reserved.iter();
        for (index, &flag) in holds_value.iter().enumerate() {
            if index % 7 == 0
                && let Some(&additional) = reservations.next()
            {
                gaps.reserve(additional);
            }
            gaps.push(flag);
        }
        let validity = gaps.finish();
        if holds_value.iter().all(|&flag| flag) {
            assert!(validity.is_none(), "{holds_value:?} after {reserved:?}");
            return;
        }
        let validity = validity.expect("an empty slot leaves a bitmap");
        assert_eq!(validity.len(), holds_value.len());
        for (index, &flag) in holds_value.iter().enumerate() {
            assert_eq!(
                validity.get_bit(index),
                flag,
                "slot {index} of {holds_value:?} after {reserved:?}"
            );
        }
    }

    #[test]
    fn empty_slots_are_recorded_whatever_room_was_reserved() {
        let patterns: Vec<Vec<bool>> = vec![
            vec![],
            vec![true],
            vec![false],
            vec![true; 40],
            vec![false; 40],
            (0..40).map(|index| index % 3 != 0).collect(),
            (0..40).map(|index| index != 39).collect(),
            (0..40).map(|index| index != 0).collect(),
            (0..200).map(|index| index % 64 != 63).collect(),
        ];
        // Nothing reserved, exactly enough, too little, too much, and in parts.
        let reservations: [&[usize]; 6] = [&[], &[40], &[8], &[1000], &[3, 5, 9], &[0, 0, 200]];
        for pattern in &patterns {
            for reserved in reservations {
                check(pattern, reserved);
            }
        }
    }
}
