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
//! - [`dataframe::Decimal128Encode`]: how a decimal type reports its `i128`
//!   mantissa at a target scale. The `rust_decimal` feature (on by default)
//!   provides the impl for `rust_decimal::Decimal`.
//!
//! The `chrono` feature (on by default) provides the `chrono` date and time
//! impls.

// `polars` pulls a wide transitive dependency tree (ahash, foldhash,
// hashbrown, windows-sys variants, …) where multiple resolved versions are
// unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables, and it would fire ~21 times on
// dependencies entirely outside this crate's control. Allow it here so the
// lint surface stays focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

pub mod dataframe;
