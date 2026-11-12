//! The encoder every conversion runs through, and the error context around
//! it.
//!
//! The functions that are generic over the row type contain only what
//! depends on it: the calls into its builder. Assembling the frame, checking
//! a schema and wording an error are ordinary functions, compiled once
//! rather than once per row type.

use std::any::type_name;
use std::borrow::Borrow;

use polars::prelude::{Column, DataFrame, DataType, PlSmallStr, PolarsError, PolarsResult};

use super::{ColumnBuilder, Columnar, Fields};

/// Bytes of rows encoded per block.
///
/// Within a block every column buffer is filled by its own loop, so only one
/// buffer is being written at a time, and the block's rows are read from
/// memory once and from cache afterwards. 4 KiB is one way of a 32 KiB
/// eight-way first-level data cache, the smallest in common use, so the rows
/// of a block never compete with each other for a place in it.
///
/// Short blocks also keep each column's run of writes short, which is what
/// lets rows with dozens of columns go fast. The README's benchmark section
/// records how results vary with this value and with the one below.
const BLOCK_BYTES: usize = 4 * 1024;

/// The fewest rows in a block, however large a row is.
///
/// Starting a column's loop has a fixed cost. Below a few dozen rows per
/// block that cost shows, and timings become erratic.
const MIN_BLOCK_ROWS: usize = 32;

/// How many values of type `T` make one block.
///
/// A block is sized in bytes rather than rows so that it fits the cache
/// whatever the row type: large rows get short blocks, small rows long ones.
/// Values behind pointers (strings, lists) are not counted.
const fn block_len<T>() -> usize {
    let rows = match size_of::<T>() {
        0 => return usize::MAX,
        size => BLOCK_BYTES / size,
    };
    if rows < MIN_BLOCK_ROWS {
        MIN_BLOCK_ROWS
    } else {
        rows
    }
}

/// A builder's `schema` function, for code that is not generic over it.
type SchemaFn = fn(PlSmallStr, &mut Fields);

/// The slots of a run of rows. Every one of them holds a value.
fn slots<'a, T: 'a, R: Borrow<T>>(rows: &'a [R]) -> impl Iterator<Item = Option<&'a T>> + Clone {
    rows.iter().map(|row| Some(<R as Borrow<T>>::borrow(row)))
}

/// Encodes rows held in a slice, either by value or by reference.
///
/// The rows are measured, every buffer is allocated at its final size, and
/// the rows are appended, a block at a time in both passes.
pub(super) fn encode<T, R>(rows: &[R]) -> PolarsResult<DataFrame>
where
    T: Columnar,
    R: Borrow<T>,
{
    let mut builder = <T::Builder as ColumnBuilder<T>>::new();
    for block in rows.chunks(block_len::<T>()) {
        builder.measure(slots(block));
    }
    builder.reserve(rows.len());

    let mut encoded = 0;
    for block in rows.chunks(block_len::<T>()) {
        if let Err(error) = builder.extend(slots(block)) {
            return Err(match first_failure::<T, R, T::Builder>(block) {
                Some((row, error)) => at_row(&error, type_name::<T>(), encoded + row),
                None => unlocated(&error),
            });
        }
        encoded += block.len();
    }

    let mut columns = Vec::new();
    builder.finish(PlSmallStr::EMPTY, None, &mut columns)?;
    let schema: SchemaFn = <T::Builder as ColumnBuilder<T>>::schema;
    assemble(rows.len(), columns, type_name::<T>(), schema)
}

/// Appends every value of a slice: the elements of one list.
///
/// A single value is one slot, so it is pushed. Several are handed over a
/// block at a time, and an empty slice costs nothing.
#[inline(always)]
#[allow(clippy::inline_always)]
pub(super) fn extend_from_slice<T, B>(builder: &mut B, values: &[T]) -> PolarsResult<()>
where
    B: ColumnBuilder<T>,
{
    if let [value] = values {
        return builder.push(Some(value));
    }
    for block in values.chunks(block_len::<T>()) {
        builder.extend(block.iter().map(Some))?;
    }
    Ok(())
}

/// Finds the first of `values` that cannot be appended, and why.
///
/// `extend` reports that a block failed, not where. The values are pushed
/// again one at a time, into a builder of their own, and the first error is
/// the one a caller sees: it names the fields it passed through on the way.
#[cold]
#[inline(never)]
pub(super) fn first_failure<T, R, B>(values: &[R]) -> Option<(usize, PolarsError)>
where
    R: Borrow<T>,
    B: ColumnBuilder<T>,
{
    let mut probe = B::new();
    for (index, value) in values.iter().enumerate() {
        if let Err(error) = probe.push(Some(<R as Borrow<T>>::borrow(value))) {
            return Some((index, error));
        }
    }
    None
}

/// Returns a row type's column names and dtypes.
pub(super) fn schema<T: Columnar>() -> PolarsResult<Vec<(String, DataType)>> {
    let schema: SchemaFn = <T::Builder as ColumnBuilder<T>>::schema;
    Ok(checked_fields(type_name::<T>(), schema)?
        .into_iter()
        .map(|(name, dtype)| (name.into_string(), dtype))
        .collect())
}

/// Collects a row type's schema and rejects one that names a column twice.
fn checked_fields(row_type: &str, schema: SchemaFn) -> PolarsResult<Fields> {
    let mut fields = Fields::new();
    schema(PlSmallStr::EMPTY, &mut fields);
    let mut seen = std::collections::HashSet::with_capacity(fields.len());
    for (name, _) in &fields {
        if !seen.insert(name.as_str()) {
            return Err(PolarsError::ComputeError(
                format!(
                    "df-derive: duplicate column `{name}` in the DataFrame schema of {row_type}"
                )
                .into(),
            ));
        }
    }
    Ok(fields)
}

/// Makes a frame of finished columns.
fn assemble(
    height: usize,
    columns: Vec<Column>,
    row_type: &str,
    schema: SchemaFn,
) -> PolarsResult<DataFrame> {
    if columns.is_empty() {
        return Ok(DataFrame::empty_with_height(height));
    }
    DataFrame::new(height, columns).map_err(|error| {
        // Polars rejects duplicate column names here. Report that the way
        // `schema` does, which names the column and the row type.
        if matches!(error, PolarsError::Duplicate(_))
            && let Err(duplicate) = checked_fields(row_type, schema)
        {
            return duplicate;
        }
        error
    })
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

/// Separates the path from the message while an error travels up through
/// nested builders. Replaced before the error reaches the caller.
const PATH_END: char = '\u{1f}';

/// Puts `step` in front of the path an error has collected so far. A list
/// index attaches to the step before it; anything else is joined by a dot.
#[cold]
#[inline(never)]
fn prefixed(error: &PolarsError, step: &dyn std::fmt::Display) -> PolarsError {
    error.wrap_msg(|message| {
        if !message.contains(PATH_END) {
            return format!("{step}{PATH_END}{message}");
        }
        let dot = if message.starts_with('[') { "" } else { "." };
        format!("{step}{dot}{message}")
    })
}

/// Replaces the separator with the colon a caller sees.
#[cold]
#[inline(never)]
fn unlocated(error: &PolarsError) -> PolarsError {
    error.wrap_msg(|message| message.replacen(PATH_END, ": ", 1))
}

/// Prefixes an encoding error with the field it came from.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn in_field(error: PolarsError, field: &'static str) -> PolarsError {
    prefixed(&error, &field)
}

/// Prefixes an encoding error with the position of the list element it came
/// from.
#[allow(clippy::needless_pass_by_value)]
pub(super) fn in_element(error: PolarsError, index: usize) -> PolarsError {
    prefixed(&error, &format_args!("[{index}]"))
}

/// Finishes an error's context: `Type[row].path: message`.
///
/// `row_type` is the type's full name. Its last path segment is what is
/// shown, with any generic arguments.
fn at_row(error: &PolarsError, row_type: &str, row: usize) -> PolarsError {
    let head = row_type.split('<').next().unwrap_or(row_type);
    let short = &row_type[head.rfind("::").map_or(0, |index| index + 2)..];
    unlocated(&prefixed(error, &format_args!("{short}[{row}]")))
}

/// Turns a failed `try_via` conversion into an encoding error.
#[cold]
#[inline(never)]
#[must_use]
pub fn conversion_error(error: &dyn std::fmt::Display) -> PolarsError {
    PolarsError::ComputeError(error.to_string().into())
}
