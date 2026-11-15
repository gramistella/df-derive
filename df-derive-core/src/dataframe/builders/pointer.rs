use std::borrow::Cow;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;

use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};

/// Encodes a pointer by encoding the value it points to.
///
/// `Box<T>`, `Rc<T>`, `Arc<T>`, `&T` and `Cow<'_, T>` are transparent: they
/// produce exactly the columns `T` produces.
pub struct PtrBuilder<B>(B);

/// The slot of the value a pointer in a slot points to.
#[inline(always)]
fn target<P: Deref>(slot: Option<&P>) -> Option<&P::Target> {
    slot.map(Deref::deref)
}

macro_rules! transparent_pointer {
    ($(<$($lt:lifetime,)? $t:ident> $ptr:ty $(where $bound:path)?;)*) => {$(
        impl<$($lt,)? $t, B> ColumnBuilder<$ptr> for PtrBuilder<B>
        where
            $t: ?Sized $(+ $bound)?,
            B: ColumnBuilder<$t>,
        {
            const ROW_AT_A_TIME: bool = B::ROW_AT_A_TIME;
            const COLUMNS: usize = B::COLUMNS;
            const MEASURES: bool = B::MEASURES;

            fn schema(name: PlSmallStr, out: &mut Fields) {
                B::schema(name, out);
            }

            fn new() -> Self {
                Self(B::new())
            }

            #[inline(always)]
            fn measure<'r, I>(&mut self, values: I)
            where
                $ptr: 'r,
                I: Iterator<Item = Option<&'r $ptr>> + Clone,
            {
                self.0.measure(values.map(target));
            }

            fn reserve(&mut self, additional: usize) {
                self.0.reserve(additional);
            }

            #[inline(always)]
            fn push(&mut self, value: Option<&$ptr>) -> PolarsResult<()> {
                self.0.push(target(value))
            }

            #[inline(always)]
            fn extend<'r, I>(&mut self, values: I) -> PolarsResult<()>
            where
                $ptr: 'r,
                I: Iterator<Item = Option<&'r $ptr>> + Clone,
            {
                self.0.extend(values.map(target))
            }

            fn finish(
                self,
                name: PlSmallStr,
                validity: Option<&Bitmap>,
                out: &mut Vec<Column>,
            ) -> PolarsResult<()> {
                self.0.finish(name, validity, out)
            }
        }

        impl<$($lt,)? $t> Columnar for $ptr
        where
            $t: Columnar + ?Sized $(+ $bound)?,
        {
            type Builder = PtrBuilder<$t::Builder>;
        }

        impl<$($lt,)? $t> Record for $ptr
        where
            $t: Record + ?Sized $(+ $bound)?,
        {
        }
    )*};
}

transparent_pointer! {
    <T> Box<T>;
    <T> Rc<T>;
    <T> Arc<T>;
    <'a, T> &'a T;
    <'a, T> Cow<'a, T> where ToOwned;
}
