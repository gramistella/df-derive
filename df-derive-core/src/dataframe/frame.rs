//! The encoder every conversion runs through, and the error context around
//! it. Only the calls into a builder are generic over the row type; the rest
//! is compiled once.

use std::any::type_name;
use std::borrow::Borrow;

use polars::prelude::{Column, DataFrame, DataType, PlSmallStr, PolarsError, PolarsResult};

use super::{ColumnBuilder, Columnar, Fields};

/// Bytes of rows encoded per block.
///
/// 4 KiB is one way of a 32 KiB eight-way first-level data cache, the
/// smallest in common use, so the rows of a block do not evict each other.
/// The README records how results vary with this value and the one below.
const BLOCK_BYTES: usize = 4 * 1024;

/// The fewest rows in a block. Below a few dozen, the fixed cost of starting
/// each column's loop shows and timings become erratic.
const MIN_BLOCK_ROWS: usize = 32;

/// How many values of type `T` make one block. Values behind pointers
/// (strings, lists) are not counted.
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

/// Encodes rows held in a slice, by value or by reference: measure, reserve,
/// append, a block at a time.
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

/// Appends the values of a slice, such as the elements of one list. A single
/// value is pushed; more are handed over a block at a time.
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

/// The ways of a first-level data cache, in the smallest one in common use.
const CACHE_WAYS: usize = 8;

/// Whether a run of a projected type's values is pushed a row at a time,
/// straight from the conversion.
///
/// A push writes to every column buffer of the row. Buffers that grow in step
/// share a cache set, so up to `CACHE_WAYS` columns are safe to push. A wider
/// row is made into a buffer first and appended in runs.
const fn pushes_through<R, B: ColumnBuilder<R>>() -> bool {
    B::ROW_AT_A_TIME || B::COLUMNS <= CACHE_WAYS
}

/// Whether the builder of a type projected to `R` measures.
///
/// `B` has to count something, and making each row a second time has to be
/// cheap: a row that owns nothing is, one that owns heap data was allocated
/// for.
#[must_use]
pub const fn measures_through<R, B: ColumnBuilder<R>>() -> bool {
    B::MEASURES && !std::mem::needs_drop::<R>()
}

/// Empties `buffer` and retypes it, keeping its allocation. `T` and `U` are
/// one row type with different lifetimes.
///
/// Works because `Vec` collects in place when the two layouts match. If std
/// stopped doing that, this would allocate and nothing else would change.
#[inline]
fn recycle<T, U>(mut buffer: Vec<T>) -> Vec<U> {
    buffer.clear();
    buffer.into_iter().map(|_| unreachable!()).collect()
}

/// `measure` of a projected type's builder.
#[inline(always)]
#[allow(clippy::inline_always)]
pub fn measure_through<'a, T, R, B, I>(
    builder: &mut B,
    values: I,
    convert: impl Fn(&'a T) -> PolarsResult<R>,
) where
    T: 'a,
    B: ColumnBuilder<R>,
    I: Iterator<Item = Option<&'a T>>,
{
    if !measures_through::<R, B>() {
        return;
    }
    for value in values.flatten() {
        // A value that cannot be converted is reported when it is appended.
        if let Ok(row) = convert(value) {
            builder.measure(std::iter::once(Some(&row)));
        }
    }
}

/// `push` of a projected type's builder: appends `value` as the row
/// `convert` makes of it.
///
/// # Errors
/// Returns the conversion's error, or the row builder's.
#[inline(always)]
#[allow(clippy::inline_always)]
pub fn push_through<'a, T, R, B>(
    builder: &mut B,
    value: Option<&'a T>,
    convert: impl Fn(&'a T) -> PolarsResult<R>,
) -> PolarsResult<()>
where
    B: ColumnBuilder<R>,
{
    let row = match value {
        Some(value) => Some(convert(value)?),
        None => None,
    };
    builder.push(row.as_ref())
}

/// Why the rows of a block stopped being made.
enum Stopped {
    /// The block is full, or the values ran out.
    Not,
    /// The next slot is empty.
    AtEmptySlot,
    /// The next value could not be converted.
    AtFailure(PolarsError),
}

/// `extend` of a projected type's builder.
///
/// `kept` is the builder's buffer for the rows of a block. Between calls it
/// is typed `K`, the row type with `'static` lifetimes, because the
/// builder's type cannot name what the rows borrow.
///
/// # Errors
/// Returns the conversion's error, or the row builder's.
#[inline(always)]
#[allow(clippy::inline_always)]
pub fn extend_through<'a, T, R, K, B, I>(
    builder: &mut B,
    kept: &mut Vec<K>,
    mut values: I,
    convert: impl Fn(&'a T) -> PolarsResult<R>,
) -> PolarsResult<()>
where
    T: 'a,
    B: ColumnBuilder<R>,
    I: Iterator<Item = Option<&'a T>>,
{
    if pushes_through::<R, B>() {
        for value in values {
            push_through(builder, value, &convert)?;
        }
        return Ok(());
    }
    let block = block_len::<R>();
    let mut rows: Vec<R> = recycle(std::mem::take(kept));
    loop {
        let mut stopped = Stopped::Not;
        rows.extend(values.by_ref().take(block).map_while(|value| {
            let Some(value) = value else {
                stopped = Stopped::AtEmptySlot;
                return None;
            };
            convert(value)
                .map_err(|error| stopped = Stopped::AtFailure(error))
                .ok()
        }));
        extend_from_slice(builder, &rows)?;
        match stopped {
            Stopped::AtFailure(error) => return Err(error),
            Stopped::AtEmptySlot => builder.push(None)?,
            Stopped::Not if rows.len() < block => break,
            Stopped::Not => {}
        }
        rows.clear();
    }
    *kept = recycle(rows);
    Ok(())
}

/// Finds the first of `values` that cannot be appended, and why, by pushing
/// them one at a time into a builder of their own. `extend` only reports
/// that a block failed.
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
///
/// A builder that cannot make its column's type names an unknown one.
/// `empty` then encodes no rows, which fails and says why.
pub(super) fn schema<T: Columnar>(
    empty: fn() -> PolarsResult<DataFrame>,
) -> PolarsResult<Vec<(String, DataType)>> {
    let schema: SchemaFn = <T::Builder as ColumnBuilder<T>>::schema;
    let fields = checked_fields(type_name::<T>(), schema)?;
    // `is_known` looks inside lists and structs, so an unknown type is seen
    // however deep it sits.
    if !fields.iter().all(|(_, dtype)| dtype.is_known()) {
        empty()?;
    }
    Ok(fields
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
        // Polars rejects duplicate names here. Report it as `schema` does.
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
/// nested builders. Replaced before the error reaches the caller. A message
/// that holds this control character itself has its start taken for path.
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

/// Finishes an error's context: `Type[row].path: message`, with the type's
/// name cut to its last path segment.
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

#[cfg(test)]
mod tests {
    use super::recycle;

    /// Pins what `recycle` relies on: the standard library collects in place.
    #[test]
    fn a_recycled_buffer_keeps_its_allocation() {
        let text = String::from("borrowed");
        let mut buffer: Vec<&str> = Vec::with_capacity(64);
        buffer.push(text.as_str());
        let address = buffer.as_ptr().addr();

        let recycled: Vec<&'static str> = recycle(buffer);
        assert_eq!(recycled.len(), 0);
        assert_eq!(recycled.capacity(), 64);
        assert_eq!(recycled.as_ptr().addr(), address);
    }
}
