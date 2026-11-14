//! Builders for the two column types that hold text as codes.
//!
//! A `Categorical` column takes whatever strings its rows hold and gives each
//! distinct one a code as it turns up. An `Enum` column has its strings
//! listed in its type: the names of the variants of a Rust type, which are
//! known before the first row is read.
//!
//! A type that names its variants can be stored either way. The list in the
//! type of an `Enum` column is the whole of what the column may hold, so it
//! suits a type that will never be given another variant. One that may be is
//! a `Categorical` column of the same names.

use std::any::{TypeId, type_name};
use std::collections::BTreeMap;
use std::fmt::{Display, Write as _};
use std::marker::PhantomData;
use std::sync::{Arc, PoisonError, RwLock, Weak};

use polars::prelude::{
    Categorical32Type, CategoricalChunked, CategoricalMapping, Categories, ChunkAgg, Column,
    DataType, FrozenCategories, IdxCa, IdxSize, IntoSeries, NamedFrom, PlSmallStr, PolarsError,
    PolarsResult, Series, UInt32Chunked, UnknownKind,
};
use polars_arrow::array::PrimitiveArray;
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
///
/// The builder is part of the `categorical` feature, which is off by default.
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
///
/// The builder is part of the `categorical` feature, which is off by default.
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

/// The positions of a column's values among the names of their type: what
/// either builder of a type that names its variants collects.
///
/// A column of `T` is then a selection, by position, of the rows of a column
/// that holds every value of `T` once. Polars selects rows without looking
/// at the names again. A column assembled from codes has every code checked
/// against them, which was measured at three to four times the time for 100k
/// rows.
struct Positions<T> {
    /// What each slot stores: the position of its value as the index of a
    /// row in the column of variants, or a zero as the placeholder of an
    /// empty slot.
    rows: Vec<IdxSize>,
    /// The type the positions are positions in.
    of: PhantomData<fn(&T)>,
}

impl<T: Variants> Positions<T> {
    const fn new() -> Self {
        Self {
            rows: Vec::new(),
            of: PhantomData,
        }
    }

    /// What a slot stores. A position too large to be an index is stored as
    /// the largest index. No column has a row there, so it stays the
    /// position of no name.
    #[inline(always)]
    fn of_slot(slot: Option<&T>) -> IdxSize {
        slot.map_or(0, |value| {
            IdxSize::try_from(value.position()).unwrap_or(IdxSize::MAX)
        })
    }

    /// The column: the rows of `variants` at these positions.
    fn select(
        self,
        variants: &Series,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        // Positions select rows as they are. Where some slots are empty,
        // they first become a column that says which. Either way Polars
        // checks that every position is that of a row, and so of a name.
        let column = if validity.is_some() {
            let positions = primitive_array(self.rows, validity)?;
            let positions = IdxCa::with_chunk(PlSmallStr::EMPTY, positions);
            variants
                .take(&positions)
                .map_err(|error| no_such_variant::<T>(error, positions.max(), variants.len()))
        } else {
            let positions = &self.rows;
            variants.take_slice(positions).map_err(|error| {
                no_such_variant::<T>(error, positions.iter().copied().max(), variants.len())
            })
        }?;
        out.push(column.with_name(name).into());
        Ok(())
    }
}

/// The names of each type stored through [`Variants`] so far, as text.
static NAMES: RwLock<BTreeMap<TypeId, Arc<[Box<str>]>>> = RwLock::new(BTreeMap::new());

/// The names of `T` as text, in the order `T` gives them.
///
/// They are asked for once for a type and kept: the names of a type do not
/// change, and formatting them for every column would cost more than a small
/// frame does. Names that repeat or cannot be written are an error here, so
/// they are one for either kind of column.
fn names<T: Variants>() -> PolarsResult<Arc<[Box<str>]>> {
    let key = TypeId::of::<T>();
    let kept = NAMES.read().unwrap_or_else(PoisonError::into_inner);
    if let Some(names) = kept.get(&key) {
        return Ok(Arc::clone(names));
    }
    drop(kept);

    let mut names = Vec::new();
    for name in T::names() {
        let mut text = String::new();
        if write!(text, "{name}").is_err() {
            return Err(unnamed::<T>());
        }
        names.push(text.into_boxed_str());
    }
    let mut sorted: Vec<&str> = names.iter().map(|name| &**name).collect();
    sorted.sort_unstable();
    if let Some(twice) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(repeated::<T>(twice[0]));
    }

    let names: Arc<[Box<str>]> = names.into();
    NAMES
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, Arc::clone(&names));
    Ok(names)
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

/// The error for a name that two values of a type have.
#[cold]
#[inline(never)]
fn repeated<T: Variants>(name: &str) -> PolarsError {
    PolarsError::ComputeError(
        format!(
            "df-derive: two values of `{}` are named `{name}`: a name stands for one value only",
            type_name::<T>()
        )
        .into(),
    )
}

/// The error for a position that is the position of no name.
#[cold]
#[inline(never)]
fn past_the_names<T: Variants>(position: IdxSize, names: usize) -> PolarsError {
    // A position too large to be stored was stored as the largest there is.
    let or_more = if position == IdxSize::MAX {
        " or more"
    } else {
        ""
    };
    PolarsError::ComputeError(
        format!(
            "df-derive: `{}::position` returned {position}{or_more}, but `names` gives {names} \
             names",
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
    past_the_names::<T>(largest, names)
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
/// value outside its list is text, not an enum, and one whose list may be
/// added to later is better stored by a [`CategoricalVariantsBuilder`]: the
/// list is the type of the column, and Polars does not put two `Enum`
/// columns together whose lists differ.
///
/// The builder is part of the `categorical` feature, which is off by default.
pub struct EnumBuilder<T>(Positions<T>);

/// The column [`variants`] made for each type stored in an `Enum` column so
/// far.
static KNOWN: RwLock<BTreeMap<TypeId, Series>> = RwLock::new(BTreeMap::new());

/// Every value of `T` once, in the order of its names, as an `Enum` column:
/// what an `Enum` column of `T` is selected from.
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

    let names = names::<T>()?;
    let names: Vec<&str> = names.iter().map(|name| &**name).collect();
    let categories = FrozenCategories::new(names.iter().copied())?;
    let variants = Series::new(PlSmallStr::EMPTY, names)
        .cast(&DataType::from_frozen_categories(categories))?;
    KNOWN
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, variants.clone());
    Ok(variants)
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
        Self(Positions::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.0.rows.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&T>) -> PolarsResult<()> {
        self.0.rows.push(Positions::of_slot(value));
        Ok(())
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        T: 'a,
        I: Iterator<Item = Option<&'a T>> + Clone,
    {
        self.0.rows.extend(values.map(Positions::of_slot));
        Ok(())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.0.select(&variants::<T>()?, name, validity, out)
    }
}

/// Builder for a `Categorical` column from a type that names its variants.
///
/// This is the builder `#[derive(ToDataFrame)]` gives an enum whose variants
/// hold no data when the enum asks with `#[df_derive(categorical)]`, and the
/// one to name for a type that implements [`Variants`] by hand and may be
/// given another value later.
///
/// The type of an `Enum` column is the list of its names, and Polars does
/// not stack, compare or join two `Enum` columns whose lists differ. A name
/// added to a type would therefore change the type of its columns, and a
/// frame saved before could not be used with one built after. A
/// `Categorical` column lists nothing in its type, so its type stays what it
/// is, and it shares Polars's global set of categories with every other such
/// column.
///
/// As for an [`EnumBuilder`], a row is the [`position`](Variants::position)
/// of its value, and nothing is formatted to store one.
///
/// The builder is part of the `categorical` feature, which is off by default.
pub struct CategoricalVariantsBuilder<T>(Positions<T>);

/// The codes that the names of a type have in one set of categories.
struct Codes {
    /// The set the codes are codes in. Polars keeps its global set for as
    /// long as a column uses it and starts another one afterwards, so the
    /// codes are good for as long as this is the set in use.
    mapping: Weak<CategoricalMapping>,
    /// The code of each name, in the order the type gives the names.
    by_position: PrimitiveArray<u32>,
}

/// The codes last given to the names of each type stored in a `Categorical`
/// column through [`Variants`].
static CODES: RwLock<BTreeMap<TypeId, Arc<Codes>>> = RwLock::new(BTreeMap::new());

/// Every value of `T` once, in the order of its names, as a `Categorical`
/// column of Polars's global set of categories: what a longer column of `T`
/// is selected from.
///
/// The codes of the names are kept from one column to the next, so the names
/// are looked up among the categories once for as long as the set lives, not
/// once for every column. The column itself is made each time. Kept, it
/// would hold on to the set, and Polars could never let the set go.
fn named<T: Variants>() -> PolarsResult<Series> {
    let categories = Categories::global();
    let mapping = categories.mapping();
    let key = TypeId::of::<T>();
    let kept = CODES
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
        // The kept codes hold on to the allocation of their set, so no other
        // set can be at its address.
        .filter(|codes| std::ptr::eq(codes.mapping.as_ptr(), Arc::as_ptr(&mapping)))
        .cloned();
    let codes = if let Some(codes) = kept {
        codes
    } else {
        let by_position = names::<T>()?
            .iter()
            .map(|name| mapping.insert_cat(name))
            .collect::<PolarsResult<Vec<u32>>>()?;
        let codes = Arc::new(Codes {
            mapping: Arc::downgrade(&mapping),
            by_position: PrimitiveArray::from_vec(by_position),
        });
        CODES
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, Arc::clone(&codes));
        codes
    };

    let codes = UInt32Chunked::with_chunk(PlSmallStr::EMPTY, codes.by_position.clone());
    let dtype = DataType::Categorical(categories, mapping);
    let column = CategoricalChunked::<Categorical32Type>::from_cats_and_dtype(codes, dtype);
    Ok(column.into_series())
}

impl<T: Variants> ColumnBuilder<T> for CategoricalVariantsBuilder<T> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        // A type whose names are refused has no column, of either kind. As
        // for an `Enum` column, `finish` says why, and so does a row type
        // asked for its schema.
        let dtype = names::<T>().map_or(DataType::Unknown(UnknownKind::Any), |_| {
            CategoricalBuilder::dtype()
        });
        out.push((name, dtype));
    }

    fn new() -> Self {
        Self(Positions::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.0.rows.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, value: Option<&T>) -> PolarsResult<()> {
        self.0.rows.push(Positions::of_slot(value));
        Ok(())
    }

    #[inline(always)]
    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        T: 'a,
        I: Iterator<Item = Option<&'a T>> + Clone,
    {
        self.0.rows.extend(values.map(Positions::of_slot));
        Ok(())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        // Polars checks every code of a categorical column it is handed, so
        // it is handed the shorter of two things. A column of more rows than
        // the type has names is selected from a column of the names, which
        // is checked once however many rows are selected.
        let names = names::<T>()?;
        if self.0.rows.len() > names.len() {
            return self.0.select(&named::<T>()?, name, validity, out);
        }

        // A column of no more rows than that is handed as it is: the code of
        // each row, found by its name as the code of any text is.
        let mut values = CategoricalBuilder::global();
        values.codes.reserve(self.0.rows.len());
        for (slot, position) in self.0.rows.into_iter().enumerate() {
            if validity.is_some_and(|validity| !validity.get_bit(slot)) {
                values.push_empty();
                continue;
            }
            let Some(text) = usize::try_from(position)
                .ok()
                .and_then(|position| names.get(position))
            else {
                return Err(past_the_names::<T>(position, names.len()));
            };
            values.push_str(text)?;
        }
        values.finish_column(name, validity, out)
    }
}
