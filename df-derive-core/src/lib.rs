//! Shared runtime trait identity for `df-derive`.
//!
//! # What this crate provides
//!
//! `df-derive-core` owns the default `dataframe` traits used by the
//! user-facing `df-derive` facade. Sharing these traits across crates lets
//! derived models compose as nested `ToDataFrame` types without each crate
//! inventing a local runtime identity.
//!
//! The [`dataframe`] module exposes:
//!
//! - [`dataframe::ToDataFrame`] — the per-instance API the derive populates.
//! - [`dataframe::Columnar`] — the columnar batch API the derive populates.
//! - [`dataframe::ToDataFrameVec`] — the slice extension trait that routes
//!   `[T]::to_dataframe()` through `Columnar` or `empty_dataframe`.
//! - [`dataframe::Decimal128Encode`] — the contract for encoding a decimal
//!   value as an `i128` mantissa rescaled to a target scale. The reference
//!   `rust_decimal::Decimal` impl is gated behind the `rust_decimal`
//!   feature (enabled by default).
//! - `impl ToDataFrame for ()` and `impl Columnar for ()` — the zero-column
//!   payload behavior used by generic `Wrapper<()>` shapes.
//!
//! # When to use this crate
//!
//! Most users get this crate through `df-derive`. Depend on this crate
//! directly when you want the shared traits without the facade, or when you
//! use `df-derive-macros` directly and still want the default runtime
//! identity.
//!
//! ```toml
//! [dependencies]
//! df-derive-core = "0.5"
//! df-derive-macros = "0.5"
//! polars = "0.55"
//! ```
//!
//! Default-runtime generated code uses hidden dependency re-exports from this
//! crate, so direct `polars-arrow` dependencies are not required unless you
//! use a custom runtime.
//!
//! ```ignore
//! use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
//! use df_derive_macros::ToDataFrame;
//!
//! #[derive(ToDataFrame)]
//! struct Trade { symbol: String, price: f64, size: u64 }
//! ```
//!
//! # Validating a custom decimal backend
//!
//! The `Decimal128Encode` contract requires round-half-to-even (banker's
//! rounding) on scale-down. The reference `rust_decimal::Decimal` impl in
//! this crate honours that contract and is checked against Polars' decimal
//! string-cast behavior in this crate's integration tests.

// `polars` pulls a wide transitive dependency tree (ahash, foldhash,
// hashbrown, windows-sys variants, …) where multiple resolved versions are
// unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables, and it would fire ~21 times on
// dependencies entirely outside this crate's control. Allow it here so the
// lint surface stays focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

pub mod dataframe;
