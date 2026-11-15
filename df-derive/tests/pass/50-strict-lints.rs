//! The derive's output under the lints a strict library turns on.
//!
//! Everything public here is documented and implements `Debug`, so only
//! generated code could trip a lint. The generated builder is a hidden public
//! struct: it passes `missing_docs` because `#[doc(hidden)]` satisfies that
//! lint.
#![deny(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    unreachable_pub,
    unnameable_types,
    unused_qualifications,
    unused_results,
    unused_lifetimes,
    single_use_lifetimes,
    elided_lifetimes_in_paths,
    trivial_casts,
    trivial_numeric_casts,
    explicit_outlives_requirements,
    redundant_lifetimes,
    let_underscore_drop,
    unit_bindings,
    unsafe_code
)]

use std::collections::BTreeMap;

use df_derive::ToDataFrame;

/// A flat row of the common field types.
#[derive(ToDataFrame, Debug, Clone)]
pub struct Plain {
    /// An integer.
    pub id: u64,
    /// Text.
    pub name: String,
    /// An optional float.
    pub price: Option<f64>,
    /// A list.
    pub tags: Vec<String>,
    /// A map.
    pub extra: BTreeMap<String, i64>,
    /// A tuple.
    pub pair: (i32, bool),
}

/// A row with nothing in it.
#[derive(ToDataFrame, Debug, Clone, Copy)]
pub struct Unit;

/// A tuple struct.
#[derive(ToDataFrame, Debug, Clone, Copy)]
pub struct Point(
    /// Across.
    pub f64,
    /// Up.
    pub f64,
);

/// A row that borrows.
#[derive(ToDataFrame, Debug, Clone, Copy)]
pub struct Borrowed<'a> {
    /// Borrowed text.
    pub name: &'a str,
    /// Borrowed bytes.
    #[df_derive(as_binary)]
    pub raw: &'a [u8],
}

/// A generic row.
#[derive(ToDataFrame, Debug, Clone)]
pub struct Generic<T> {
    /// The payload.
    pub value: T,
    /// A count.
    pub n: u32,
}

/// A const parameter before a type parameter.
#[derive(ToDataFrame, Debug, Clone)]
pub struct Mixed<const N: usize, T> {
    /// Items.
    pub items: [T; N],
}

/// A value that takes the name of the field that holds it.
#[derive(ToDataFrame, Debug, Clone, Copy)]
#[df_derive(transparent)]
pub struct Meters(
    /// The length.
    pub f64,
);

/// A type stored as a row that borrows from it.
#[derive(ToDataFrame, Debug, Clone)]
#[df_derive(via = "Borrowed<'_>")]
pub struct Projected {
    /// The name.
    pub name: String,
    /// The bytes.
    pub raw: Vec<u8>,
}

impl<'a> From<&'a Projected> for Borrowed<'a> {
    fn from(value: &'a Projected) -> Self {
        Self {
            name: &value.name,
            raw: &value.raw,
        }
    }
}

/// A generic type stored as a generic row.
#[derive(ToDataFrame, Debug, Clone)]
#[df_derive(via = "Generic<T>")]
pub struct Wrapped<T: Clone> {
    /// The value.
    pub inner: T,
}

impl<T: Clone> From<&Wrapped<T>> for Generic<T> {
    fn from(value: &Wrapped<T>) -> Self {
        Self {
            value: value.inner.clone(),
            n: 0,
        }
    }
}

/// A row that holds the others.
#[derive(ToDataFrame, Debug, Clone)]
pub struct Order {
    /// A wrapped value.
    pub length: Meters,
    /// A nested row, spliced in.
    #[df_derive(flatten)]
    pub point: Point,
    /// A formatted value.
    #[df_derive(as_string)]
    pub code: u8,
    /// Nested rows.
    pub legs: Vec<Generic<i64>>,
    /// A projected value.
    pub projected: Option<Projected>,
}

fn main() {}
