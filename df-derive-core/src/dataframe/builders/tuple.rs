use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;

use crate::dataframe::frame::{child, in_field};
use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Builder for `()`, which contributes no columns.
pub struct UnitBuilder;

impl ColumnBuilder<()> for UnitBuilder {
    fn schema(_name: PlSmallStr, _out: &mut Fields) {}

    fn with_capacity(_rows: usize) -> Self {
        Self
    }

    #[inline(always)]
    fn push(&mut self, _value: &()) -> PolarsResult<()> {
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {}

    fn finish(
        self,
        _name: PlSmallStr,
        _validity: Option<&Bitmap>,
        _out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        Ok(())
    }
}

impl Columnar for () {
    type Builder = UnitBuilder;
}

// A tuple is a struct whose fields are named `field_0`, `field_1`, ...
macro_rules! tuple {
    ($(($($index:tt $t:ident $b:ident $field:literal),+))*) => {$(
        impl<$($t, $b: ColumnBuilder<$t>),+> ColumnBuilder<($($t,)+)> for ($($b,)+) {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                $($b::schema(child(&name, $field), out);)+
            }

            const LISTS: usize = 0 $(+ $b::LISTS)+;

            fn with_capacity(rows: usize) -> Self {
                ($($b::with_capacity(rows),)+)
            }

            #[inline(always)]
            #[allow(unused_assignments)]
            fn measure(value: &($($t,)+), counts: &mut [usize]) {
                let mut at = 0usize;
                $(
                    if $b::LISTS > 0 {
                        $b::measure(&value.$index, &mut counts[at..at + $b::LISTS]);
                    }
                    at += $b::LISTS;
                )+
            }

            #[allow(unused_assignments)]
            fn with_capacities(rows: usize, counts: &[usize]) -> Self {
                let mut at = 0usize;
                ($({
                    let start = at;
                    at += $b::LISTS;
                    $b::with_capacities(rows, &counts[start..at])
                },)+)
            }

            #[inline(always)]
            fn push(&mut self, value: &($($t,)+)) -> PolarsResult<()> {
                $(
                    if let Err(error) = self.$index.push(&value.$index) {
                        return Err(in_field(error, $field));
                    }
                )+
                Ok(())
            }

            #[inline(always)]
            fn extend_from<'a, I>(&mut self, values: I) -> PolarsResult<()>
            where
                ($($t,)+): 'a,
                I: Iterator<Item = &'a ($($t,)+)> + Clone,
            {
                $(
                    self.$index
                        .extend_from(values.clone().map(|value| &value.$index))?;
                )+
                Ok(())
            }

            #[inline(always)]
            fn push_null(&mut self) {
                $(self.$index.push_null();)+
            }

            fn finish(
                self,
                name: PlSmallStr,
                validity: Option<&Bitmap>,
                out: &mut Vec<Column>,
            ) -> PolarsResult<()> {
                $(self.$index.finish(child(&name, $field), validity, out)?;)+
                Ok(())
            }
        }

        impl<$($t: Columnar),+> Columnar for ($($t,)+) {
            type Builder = ($($t::Builder,)+);
        }

        impl<$($t: Columnar),+> Record for ($($t,)+) {}
    )*};
}

tuple! {
    (0 T0 B0 "field_0")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4", 5 T5 B5 "field_5")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4", 5 T5 B5 "field_5", 6 T6 B6 "field_6")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4", 5 T5 B5 "field_5", 6 T6 B6 "field_6", 7 T7 B7 "field_7")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4", 5 T5 B5 "field_5", 6 T6 B6 "field_6", 7 T7 B7 "field_7", 8 T8 B8 "field_8")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4", 5 T5 B5 "field_5", 6 T6 B6 "field_6", 7 T7 B7 "field_7", 8 T8 B8 "field_8", 9 T9 B9 "field_9")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4", 5 T5 B5 "field_5", 6 T6 B6 "field_6", 7 T7 B7 "field_7", 8 T8 B8 "field_8", 9 T9 B9 "field_9", 10 T10 B10 "field_10")
    (0 T0 B0 "field_0", 1 T1 B1 "field_1", 2 T2 B2 "field_2", 3 T3 B3 "field_3", 4 T4 B4 "field_4", 5 T5 B5 "field_5", 6 T6 B6 "field_6", 7 T7 B7 "field_7", 8 T8 B8 "field_8", 9 T9 B9 "field_9", 10 T10 B10 "field_10", 11 T11 B11 "field_11")
}
