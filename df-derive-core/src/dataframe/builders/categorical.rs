//! Builders for the two column types that hold text as codes.
//!
//! A `Categorical` column takes whatever strings its rows hold and gives each
//! distinct one a code as it turns up. An `Enum` column has its strings
//! listed in its type: the names of the variants of a Rust type, which are
//! known before the first row is read.

use std::any::{TypeId, type_name};
use std::collections::BTreeMap;
use std::fmt::{Display, Write as _};
use std::marker::PhantomData;
use std::sync::{Arc, PoisonError, RwLock};

use polars::prelude::{
    Categorical32Type, CategoricalChunked, CategoricalMapping, Categories, ChunkAgg, Column,
    DataType, FrozenCategories, IdxCa, IdxSize, IntoSeries, NamedFrom, PlSmallStr, PolarsError,
    PolarsResult, Series, UInt32Chunked, UnknownKind,
};
use polars_arrow::bitmap::Bitmap;

use super::primitive::primitive_array;
use super::text::display_error;
use crate::dataframe::{ColumnBuilder, Fields, Variants};

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

/// Builder for an `Enum` column from a type that names its variants.
///
/// This is the builder `#[derive(ToDataFrame)]` gives an enum whose variants
/// hold no data, and the one to name for a type that implements [`Variants`]
/// by hand. The column's type lists the names of the variants, so it says
/// what the column can hold before a row is read, and every column of one
/// Rust type has the same type.
///
/// Nothing is looked up to store a value, which a [`CategoricalBuilder`]
/// does for every string: a row is the [`position`](Variants::position) of
/// its value. The price is that the names are fixed. A type that can hold a
/// value outside its list is text, not an enum.
pub struct EnumBuilder<T> {
    /// The position of each slot's value among the names.
    positions: Vec<IdxSize>,
    /// The type the positions are positions in.
    of: PhantomData<fn(&T)>,
}

/// The column [`variants`] made for each type stored in an `Enum` column so
/// far.
static KNOWN: RwLock<BTreeMap<TypeId, Series>> = RwLock::new(BTreeMap::new());

/// Every value of `T` once, in the order of its names, as an `Enum` column.
///
/// A column of `T` is a selection of rows from this one, by position. Polars
/// selects rows without looking at the names again. A column assembled from
/// codes has every code checked against them, which was measured at three
/// times the time for 100k rows.
///
/// The column is made once for a type and kept. Making it has Polars hash
/// every name and look for an equal set of names among those in use, which
/// was measured at half a microsecond: too much for every column of every
/// frame, when a frame of one row takes a microsecond in all.
fn variants<T: Variants>() -> PolarsResult<Series> {
    let key = TypeId::of::<T>();
    let known = KNOWN.read().unwrap_or_else(PoisonError::into_inner);
    if let Some(variants) = known.get(&key) {
        return Ok(variants.clone());
    }
    drop(known);

    let mut names = Vec::new();
    for name in T::names() {
        let mut text = String::new();
        if write!(text, "{name}").is_err() {
            return Err(unnamed::<T>());
        }
        names.push(text);
    }
    let categories = FrozenCategories::new(names.iter().map(String::as_str))?;
    let variants = Series::new(PlSmallStr::EMPTY, names)
        .cast(&DataType::from_frozen_categories(categories))?;
    KNOWN
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, variants.clone());
    Ok(variants)
}

/// The error for a name that could not be written as text.
#[cold]
#[inline(never)]
fn unnamed<T: Variants>() -> PolarsError {
    PolarsError::ComputeError(
        format!(
            "df-derive: a name of `{}` could not be formatted: its `Display` returned an error",
            type_name::<T>()
        )
        .into(),
    )
}

/// What a selection that failed is reported as.
///
/// Polars refuses a position that no row of the column of variants has. The
/// largest position is one of those, and says how far off the type is.
#[cold]
#[inline(never)]
fn no_such_variant<T: Variants>(
    error: PolarsError,
    largest: Option<IdxSize>,
    names: usize,
) -> PolarsError {
    let (PolarsError::OutOfBounds(_), Some(largest)) = (&error, largest) else {
        return error;
    };
    // A position too large to be stored was stored as the largest there is.
    let or_more = if largest == IdxSize::MAX {
        " or more"
    } else {
        ""
    };
    PolarsError::ComputeError(
        format!(
            "df-derive: `{}::position` returned {largest}{or_more}, but `names` gives {names} names",
            type_name::<T>()
        )
        .into(),
    )
}

/// What a slot stores: the position of its value as the index of a row in
/// the column of variants, or a zero as the placeholder of an empty slot.
///
/// A position too large to be an index is stored as the largest index. No
/// column has a row there, so it stays the position of no name.
#[inline(always)]
fn stored<T: Variants>(slot: Option<&T>) -> IdxSize {
    slot.map_or(0, |value| {
        IdxSize::try_from(value.position()).unwrap_or(IdxSize::MAX)
    })
}

impl<T: Variants> ColumnBuilder<T> for EnumBuilder<T> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        // A type whose names Polars refuses has no column type to name.
        // Nothing is stored as the one named here: `finish` says why, and
        // so does a row type asked for its schema.
        let dtype = variants::<T>().map_or(DataType::Unknown(UnknownKind::Any), |variants| {
            variants.dtype().clone()
        });
        out.push((name, dtype));
    }

    fn new() -> Self {
        Self {
            positions: Vec::new(),
            of: PhantomData,
        }
    }

    fn reserve(&mut self, additional: usize) {
        self.positions.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&T>) -> PolarsResult<()> {
        self.positions.push(stored(value));
        Ok(())
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        T: 'a,
        I: Iterator<Item = Option<&'a T>> + Clone,
    {
        self.positions.extend(values.map(stored));
        Ok(())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        let variants = variants::<T>()?;
        // Positions select rows as they are. Where some slots are empty,
        // they first become a column that says which. Either way Polars
        // checks that every position is that of a row, and so of a name.
        let column = if validity.is_some() {
            let positions = primitive_array(self.positions, validity)?;
            let positions = IdxCa::with_chunk(PlSmallStr::EMPTY, positions);
            variants
                .take(&positions)
                .map_err(|error| no_such_variant::<T>(error, positions.max(), variants.len()))
        } else {
            let positions = &self.positions;
            variants.take_slice(positions).map_err(|error| {
                no_such_variant::<T>(error, positions.iter().copied().max(), variants.len())
            })
        }?;
        out.push(column.with_name(name).into());
        Ok(())
    }
}
