use std::borrow::Cow;
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

macro_rules! transparent_pointer {
    ($(<$($lt:lifetime,)? $t:ident> $ptr:ty $(where $bound:path)?;)*) => {$(
        impl<$($lt,)? $t, B> ColumnBuilder<$ptr> for PtrBuilder<B>
        where
            $t: ?Sized $(+ $bound)?,
            B: ColumnBuilder<$t>,
        {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                B::schema(name, out);
            }

            const LISTS: usize = B::LISTS;

            fn with_capacity(rows: usize) -> Self {
                Self(B::with_capacity(rows))
            }

            #[inline(always)]
            fn measure(value: &$ptr, counts: &mut [usize]) {
                B::measure(&**value, counts);
            }

            fn with_capacities(rows: usize, counts: &[usize]) -> Self {
                Self(B::with_capacities(rows, counts))
            }

            #[inline(always)]
            fn push(&mut self, value: &$ptr) -> PolarsResult<()> {
                self.0.push(&**value)
            }

            #[inline(always)]
            fn push_null(&mut self) {
                self.0.push_null();
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
