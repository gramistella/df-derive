//! The single row loop and frame assembly shared by every conversion.

use polars::prelude::{DataFrame, DataType, PlSmallStr, PolarsError, PolarsResult};

use super::{ColumnBuilder, Columnar, Fields};

/// Separates the field path from the message while an error travels up
/// through nested builders. Replaced before the error reaches the caller.
const PATH_END: char = '\u{1f}';

/// List counters kept on the stack. A row type with at most this many nested
/// lists is measured without a heap allocation.
const INLINE_LISTS: usize = 8;

/// Rows up to this many bytes are encoded one field at a time: each pass
/// reads a small stride and fills a single column with a tight loop.
pub(super) const SMALL_ROW: usize = 64;

/// The most buffers one pass over larger rows writes to. Feeding many column
/// buffers from a single loop makes them compete for the same cache sets, so
/// wide rows are encoded a few fields per pass.
pub const PASS_STREAMS: usize = 8;

/// Encodes a slice.
///
/// Rows that contain lists are measured first so every list buffer is
/// allocated once at its final size. The rows are then appended in bulk; if
/// that fails, they are encoded again row by row, which is the path that
/// knows which row and field the error came from.
pub(super) fn encode_slice<T: Columnar>(rows: &[T]) -> PolarsResult<DataFrame> {
    let lists = <T::Builder as ColumnBuilder<T>>::LISTS;
    let mut builder = if lists == 0 {
        <T::Builder as ColumnBuilder<T>>::with_capacity(rows.len())
    } else if lists <= INLINE_LISTS {
        let mut counts = [0usize; INLINE_LISTS];
        for row in rows {
            <T::Builder as ColumnBuilder<T>>::measure(row, &mut counts[..lists]);
        }
        <T::Builder as ColumnBuilder<T>>::with_capacities(rows.len(), &counts[..lists])
    } else {
        let mut counts = vec![0usize; lists];
        for row in rows {
            <T::Builder as ColumnBuilder<T>>::measure(row, &mut counts);
        }
        <T::Builder as ColumnBuilder<T>>::with_capacities(rows.len(), &counts)
    };
    if let Err(error) = builder.extend_from(rows.iter(), super::is_narrow::<T>()) {
        return Err(locate(rows, error));
    }
    finish::<T>(builder, rows.len())
}

/// Finds the row and field of a bulk encoding error by encoding row by row.
#[cold]
#[inline(never)]
fn locate<T: Columnar>(rows: &[T], error: PolarsError) -> PolarsError {
    encode(rows).err().unwrap_or(error)
}

/// Encodes rows from any iterator in one pass. List buffers grow as needed.
pub(super) fn encode<'a, T, I>(rows: I) -> PolarsResult<DataFrame>
where
    T: Columnar + 'a,
    I: IntoIterator<Item = &'a T>,
{
    let rows = rows.into_iter();
    let mut builder = <T::Builder as ColumnBuilder<T>>::with_capacity(rows.size_hint().0);
    let mut height = 0usize;
    for row in rows {
        if let Err(error) = builder.push(row) {
            return Err(at_row::<T>(&error, height));
        }
        height += 1;
    }
    finish::<T>(builder, height)
}

fn finish<T: Columnar>(builder: T::Builder, height: usize) -> PolarsResult<DataFrame> {
    let mut columns = Vec::new();
    builder.finish(PlSmallStr::EMPTY, None, &mut columns)?;
    if columns.is_empty() {
        return Ok(DataFrame::empty_with_height(height));
    }
    DataFrame::new(height, columns).map_err(duplicate_columns::<T>)
}

pub(super) fn schema<T: Columnar>() -> PolarsResult<Vec<(String, DataType)>> {
    let mut fields = Fields::new();
    <T::Builder as ColumnBuilder<T>>::schema(PlSmallStr::EMPTY, &mut fields);
    let mut seen = std::collections::HashSet::with_capacity(fields.len());
    for (name, _) in &fields {
        if !seen.insert(name.as_str()) {
            return Err(duplicate_column::<T>(name));
        }
    }
    Ok(fields
        .into_iter()
        .map(|(name, dtype)| (name.into_string(), dtype))
        .collect())
}

/// Joins a parent column path and a field name with a dot.
#[must_use]
pub fn child(parent: &PlSmallStr, field: &'static str) -> PlSmallStr {
    if parent.is_empty() {
        return PlSmallStr::from_static(field);
    }
    let mut joined = String::with_capacity(parent.len() + 1 + field.len());
    joined.push_str(parent);
    joined.push('.');
    joined.push_str(field);
    joined.into()
}

/// Prefixes an encoding error with the field it came from.
#[cold]
#[inline(never)]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn in_field(error: PolarsError, field: &'static str) -> PolarsError {
    error.wrap_msg(|message| {
        if message.contains(PATH_END) {
            format!("{field}.{message}")
        } else {
            format!("{field}{PATH_END}{message}")
        }
    })
}

fn short_type_name<T: ?Sized>() -> &'static str {
    let full = std::any::type_name::<T>();
    let head = full.split('<').next().unwrap_or(full);
    let start = head.rfind("::").map_or(0, |index| index + 2);
    &full[start..]
}

#[cold]
#[inline(never)]
fn at_row<T: ?Sized>(error: &PolarsError, row: usize) -> PolarsError {
    let type_name = short_type_name::<T>();
    error.wrap_msg(|message| match message.split_once(PATH_END) {
        Some((path, message)) => format!("{type_name}[{row}].{path}: {message}"),
        None => format!("{type_name}[{row}]: {message}"),
    })
}

#[cold]
#[inline(never)]
fn duplicate_column<T: ?Sized>(name: &str) -> PolarsError {
    PolarsError::ComputeError(
        format!(
            "df-derive: duplicate column `{name}` in the DataFrame schema of {}",
            std::any::type_name::<T>()
        )
        .into(),
    )
}

/// Polars rejects duplicate column names when the frame is assembled. Report
/// that with the same message `schema` uses, which names the column.
#[cold]
#[inline(never)]
fn duplicate_columns<T: Columnar>(error: PolarsError) -> PolarsError {
    if matches!(error, PolarsError::Duplicate(_))
        && let Err(duplicate) = schema::<T>()
    {
        return duplicate;
    }
    error
}

/// Turns a failed `try_via` conversion into an encoding error.
#[cold]
#[inline(never)]
#[must_use]
pub fn conversion_error(error: &dyn std::fmt::Display) -> PolarsError {
    PolarsError::ComputeError(error.to_string().into())
}
