//! Runtime for [`df-derive`](https://docs.rs/df-derive).
//!
//! Most users get this crate through the `df-derive` facade, which re-exports
//! [`dataframe`] next to the derive macro. Depend on it directly only to
//! implement the traits without the macro.
//!
//! The [`dataframe`] module holds:
//!
//! - [`dataframe::Columnar`]: implemented by every type that can be stored in
//!   `DataFrame` columns. It names the type's default
//!   [`dataframe::ColumnBuilder`].
//! - [`dataframe::ColumnBuilder`]: accumulates values into one or more
//!   columns. [`dataframe::builders`] has the provided ones.
//! - [`dataframe::ToDataFrame`] and [`dataframe::ToDataFrameVec`]: the
//!   conversions users call on row types and on slices of them.
//!
//! # Features
//!
//! - `chrono` (on by default): `chrono` dates and times as field types. A
//!   `chrono::DateTime<Tz>` is an instant and becomes a `Datetime` column
//!   marked as UTC; a `NaiveDateTime` becomes one with no time zone.
//! - `rust_decimal` (on by default): `rust_decimal::Decimal` as a field
//!   type. It turns `decimal` on.
//! - `decimal`: `Decimal` columns for any decimal type. A type says how to
//!   read its `i128` mantissa at a target scale by implementing
//!   `dataframe::Decimal128Encode`, and `dataframe::builders::DecimalBuilder`
//!   stores it.
//! - `categorical`: `Categorical` columns for text that repeats, built by
//!   `dataframe::builders::CategoricalBuilder`, and columns for types that
//!   name their variants through `dataframe::Variants`: an `Enum` column
//!   built by `dataframe::builders::EnumBuilder`, or a `Categorical` one
//!   built by `dataframe::builders::CategoricalVariantsBuilder`.
//!
//! A feature also turns on the Polars features its columns need. With none
//! of them, Polars is asked only for the types that the standard library's
//! own types are stored as.

// `polars` pulls a wide transitive dependency tree (ahash, foldhash,
// hashbrown, windows-sys variants, …) where multiple resolved versions are
// unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables, and it would fire ~21 times on
// dependencies entirely outside this crate's control. Allow it here so the
// lint surface stays focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

pub mod dataframe;
