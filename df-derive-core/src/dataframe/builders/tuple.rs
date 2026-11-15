use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;

use crate::dataframe::frame::{child, in_field};
use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Builder for `()`, which contributes no columns.
pub struct UnitBuilder;

impl ColumnBuilder<()> for UnitBuilder {
    const COLUMNS: usize = 0;

    fn schema(_name: PlSmallStr, _out: &mut Fields) {}

    fn new() -> Self {
        Self
    }

    fn reserve(&mut self, _additional: usize) {}

    #[inline(always)]
    fn push(&mut self, _value: Option<&()>) -> PolarsResult<()> {
        Ok(())
    }

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
// Up to 12 elements, like the standard library's own impls for tuples.
macro_rules! tuple {
    ($(($($index:tt $t:ident $b:ident $field:literal),+))*) => {$(
        impl<$($t, $b: ColumnBuilder<$t>),+> ColumnBuilder<($($t,)+)> for ($($b,)+) {
            // A tuple is fed a row at a time when every element is.
            const ROW_AT_A_TIME: bool = true $(&& $b::ROW_AT_A_TIME)+;
            const COLUMNS: usize = 0 $(+ $b::COLUMNS)+;
            const MEASURES: bool = false $(|| $b::MEASURES)+;

            fn schema(name: PlSmallStr, out: &mut Fields) {
                $($b::schema(child(&name, $field), out);)+
            }

            fn new() -> Self {
                ($($b::new(),)+)
            }

            #[inline(always)]
            fn measure<'a, I>(&mut self, values: I)
            where
                ($($t,)+): 'a,
                I: Iterator<Item = Option<&'a ($($t,)+)>> + Clone,
            {
                $(
                    self.$index
                        .measure(values.clone().map(|value| value.map(|value| &value.$index)));
                )+
            }

            fn reserve(&mut self, additional: usize) {
                $(self.$index.reserve(additional);)+
            }

            #[inline(always)]
            fn push(&mut self, value: Option<&($($t,)+)>) -> PolarsResult<()> {
                $(
                    if let Err(error) = self.$index.push(value.map(|value| &value.$index)) {
                        return Err(in_field(error, $field));
                    }
                )+
                Ok(())
            }

            #[inline(always)]
            fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
            where
                ($($t,)+): 'a,
                I: Iterator<Item = Option<&'a ($($t,)+)>> + Clone,
            {
                // The elements that take a row at a time are fed first,
                // together, in one walk over the block.
                if false $(|| $b::ROW_AT_A_TIME)+ {
                    for value in values.clone() {
                        $(
                            if $b::ROW_AT_A_TIME {
                                self.$index.push(value.map(|value| &value.$index))?;
                            }
                        )+
                    }
                }
                // Each of the others gets the block as a run of its own.
                $(
                    if !$b::ROW_AT_A_TIME {
                        self.$index
                            .extend(values.clone().map(|value| value.map(|value| &value.$index)))?;
                    }
                )+
                Ok(())
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
