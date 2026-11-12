use std::fmt::{Display, Write as _};

use polars::prelude::{
    BinaryChunked, Column, DataType, IntoSeries, PlSmallStr, PolarsError, PolarsResult,
    StringChunked,
};
use polars_arrow::array::MutableBinaryViewArray;
use polars_arrow::bitmap::Bitmap;

use crate::dataframe::{ColumnBuilder, Columnar, Fields};

/// Builder for a string column from anything that borrows as `str`.
///
/// This is the encoding `#[df_derive(as_str)]` selects.
pub struct StrBuilder(MutableBinaryViewArray<str>);

impl StrBuilder {
    fn finish_column(self, name: PlSmallStr, validity: Option<&Bitmap>, out: &mut Vec<Column>) {
        let array = self.0.freeze().with_validity(validity.cloned());
        out.push(StringChunked::with_chunk(name, array).into_series().into());
    }
}

impl<T: AsRef<str> + ?Sized> ColumnBuilder<T> for StrBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::String));
    }

    fn with_capacity(rows: usize) -> Self {
        Self(MutableBinaryViewArray::with_capacity(rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &T) -> PolarsResult<()> {
        self.0.push_value_ignore_validity(value.as_ref());
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.0.push_value_ignore_validity("");
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.finish_column(name, validity, out);
        Ok(())
    }
}

impl Columnar for String {
    type Builder = StrBuilder;
}

impl Columnar for str {
    type Builder = StrBuilder;
}

/// Builder for a string column from anything that implements `Display`.
///
/// This is the encoding `#[df_derive(as_string)]` selects. Values are
/// formatted into one reused scratch buffer.
pub struct DisplayBuilder {
    values: StrBuilder,
    scratch: String,
}

#[cold]
#[inline(never)]
fn display_error() -> PolarsError {
    PolarsError::ComputeError(
        "df-derive: as_string Display formatting failed: an error occurred when formatting"
            .into(),
    )
}

impl<T: Display + ?Sized> ColumnBuilder<T> for DisplayBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::String));
    }

    fn with_capacity(rows: usize) -> Self {
        Self {
            values: StrBuilder(MutableBinaryViewArray::with_capacity(rows)),
            scratch: String::new(),
        }
    }

    #[inline]
    fn push(&mut self, value: &T) -> PolarsResult<()> {
        self.scratch.clear();
        if write!(self.scratch, "{value}").is_err() {
            return Err(display_error());
        }
        self.values
            .0
            .push_value_ignore_validity(self.scratch.as_str());
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.values.0.push_value_ignore_validity("");
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.values.finish_column(name, validity, out);
        Ok(())
    }
}

/// Builder for a binary column from anything that borrows as bytes.
///
/// This is the encoding `#[df_derive(as_binary)]` selects.
pub struct BinaryBuilder(MutableBinaryViewArray<[u8]>);

impl<T: AsRef<[u8]> + ?Sized> ColumnBuilder<T> for BinaryBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        out.push((name, DataType::Binary));
    }

    fn with_capacity(rows: usize) -> Self {
        Self(MutableBinaryViewArray::with_capacity(rows))
    }

    #[inline(always)]
    fn push(&mut self, value: &T) -> PolarsResult<()> {
        self.0.push_value_ignore_validity(value.as_ref());
        Ok(())
    }

    #[inline(always)]
    fn push_null(&mut self) {
        self.0.push_value_ignore_validity([0u8; 0].as_slice());
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let array = self.0.freeze().with_validity(validity.cloned());
        out.push(BinaryChunked::with_chunk(name, array).into_series().into());
        Ok(())
    }
}
