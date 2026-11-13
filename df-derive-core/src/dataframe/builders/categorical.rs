use std::fmt::{Display, Write as _};
use std::sync::Arc;

use polars::prelude::{
    Categorical32Type, CategoricalChunked, CategoricalMapping, Categories, Column, DataType,
    IntoSeries, PlSmallStr, PolarsResult, UInt32Chunked,
};
use polars_arrow::bitmap::Bitmap;

use super::primitive::primitive_array;
use super::text::display_error;
use crate::dataframe::{ColumnBuilder, Fields};

/// Builder for a `Categorical` column from anything that borrows as `str`.
///
/// This is the encoding `#[df_derive(categorical)]` selects. The column holds
/// a 32-bit code for each row and every distinct string once. The strings are
/// kept in Polars's global set of categories, which every categorical column
/// built here shares, so two such columns compare and join without one being
/// translated into the codes of the other.
///
/// It suits text that repeats: a currency, a venue, the kind of a thing. Text
/// that is different in nearly every row is better left a string column.
pub struct CategoricalBuilder {
    codes: Vec<u32>,
    categories: Arc<Categories>,
    /// The strings of `categories` and the code of each. The codes in `codes`
    /// are this mapping's, so the finished column is given this one.
    mapping: Arc<CategoricalMapping>,
}

impl CategoricalBuilder {
    /// An empty builder over the global set of categories.
    fn global() -> Self {
        let categories = Categories::global();
        let mapping = categories.mapping();
        Self {
            codes: Vec::new(),
            categories,
            mapping,
        }
    }

    /// The column type: categorical, over the global set of categories.
    fn dtype() -> DataType {
        DataType::from_categories(Categories::global())
    }

    /// Appends the code of `text`, adding it to the categories if it is new.
    #[inline(always)]
    fn push_str(&mut self, text: &str) -> PolarsResult<()> {
        self.codes.push(self.mapping.insert_cat(text)?);
        Ok(())
    }

    /// Appends the placeholder of an empty slot.
    #[inline(always)]
    fn push_empty(&mut self) {
        self.codes.push(0);
    }

    fn finish_column(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let codes = UInt32Chunked::with_chunk(name, primitive_array(self.codes, validity)?);
        let dtype = DataType::Categorical(self.categories, self.mapping);
        let column = CategoricalChunked::<Categorical32Type>::from_cats_and_dtype(codes, dtype);
        out.push(column.into_series().into());
        Ok(())
    }
}

impl<T: AsRef<str> + ?Sized> ColumnBuilder<T> for CategoricalBuilder {
    // A value is reached through a pointer and looked up with a call, as a
    // string is appended.
    const ROW_AT_A_TIME: bool = true;

    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, Self::dtype()));
    }

    fn new() -> Self {
        Self::global()
    }

    fn reserve(&mut self, additional: usize) {
        self.codes.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&T>) -> PolarsResult<()> {
        let Some(value) = value else {
            self.push_empty();
            return Ok(());
        };
        self.push_str(value.as_ref())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.finish_column(name, validity, out)
    }
}

/// Builder for a `Categorical` column from anything that implements
/// `Display`.
///
/// This is the encoding `#[df_derive(as_string, categorical)]` selects. A
/// value is formatted into one reused scratch buffer and stored as
/// [`CategoricalBuilder`] stores a string.
pub struct CategoricalDisplayBuilder {
    values: CategoricalBuilder,
    scratch: String,
}

impl<T: Display + ?Sized> ColumnBuilder<T> for CategoricalDisplayBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, CategoricalBuilder::dtype()));
    }

    fn new() -> Self {
        Self {
            values: CategoricalBuilder::global(),
            scratch: String::new(),
        }
    }

    fn reserve(&mut self, additional: usize) {
        self.values.codes.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&T>) -> PolarsResult<()> {
        let Some(value) = value else {
            self.values.push_empty();
            return Ok(());
        };
        self.scratch.clear();
        if write!(self.scratch, "{value}").is_err() {
            return Err(display_error());
        }
        self.values.push_str(self.scratch.as_str())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.values.finish_column(name, validity, out)
    }
}
