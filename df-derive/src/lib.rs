//! Derive fast conversions from Rust structs into Polars `DataFrame`s.
//!
//! ```toml
//! [dependencies]
//! df-derive = "0.6"
//! polars = "0.55"
//! ```
//!
//! ```
//! use df_derive::prelude::*;
//!
//! #[derive(ToDataFrame)]
//! struct Trade {
//!     symbol: String,
//!     price: f64,
//!     size: Option<u64>,
//! }
//!
//! # fn main() -> polars::prelude::PolarsResult<()> {
//! let trades = vec![
//!     Trade { symbol: "AAPL".into(), price: 187.23, size: Some(100) },
//!     Trade { symbol: "MSFT".into(), price: 411.61, size: None },
//! ];
//! let df = trades.as_slice().to_dataframe()?;
//! assert_eq!(df.shape(), (2, 3));
//! # Ok(())
//! # }
//! ```
//!
//! The derive targets the runtime in [`dataframe`]. Any field whose type
//! implements [`dataframe::Columnar`] works without an attribute; see the
//! derive macro's documentation for the field and container attributes, and
//! the repository README for the full list of supported types.
//!
//! A crate that re-exports this runtime under its own path tells the derive
//! where to find it with `#[df_derive(crate = "::that_crate")]`.
//!
//! # Features
//!
//! - `chrono` (on by default): `chrono` dates and times as field types.
//! - `rust_decimal` (on by default): `rust_decimal::Decimal` as a field
//!   type. It turns `decimal` on.
//! - `decimal`: `Decimal` columns for a decimal type of your own, through
//!   `dataframe::Decimal128Encode` and the `decimal(...)` field attribute.
//! - `categorical`: `Categorical` columns for text that repeats, through the
//!   `categorical` field attribute, and `Enum` or `Categorical` columns for
//!   enums whose variants hold no data, through the derive on the enum or
//!   `dataframe::Variants`.
//!
//! A feature also turns on the Polars features its columns need, so a
//! `polars` dependency needs none of them for df-derive's sake. With
//! `default-features = false`, Polars is asked only for the types that the
//! standard library's own types are stored as.

// `polars` pulls a wide transitive dependency tree where multiple resolved
// versions are unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables; allow it here so linting stays
// focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

// Lets the derive's default `::df_derive` path resolve inside this crate.
extern crate self as df_derive;

pub use df_derive_core::dataframe;
pub use df_derive_macros::ToDataFrame;

/// Common imports for normal users.
///
/// This includes the derive macro and the runtime traits. The trait
/// `ToDataFrame` is also exported as `ToDataFrameTrait` for code that wants
/// an unambiguous type-namespace name.
pub mod prelude {
    pub use crate::ToDataFrame;
    #[cfg(feature = "decimal")]
    pub use crate::dataframe::Decimal128Encode;
    #[cfg(feature = "categorical")]
    pub use crate::dataframe::Variants;
    pub use crate::dataframe::{
        ColumnBuilder, Columnar, ToDataFrame, ToDataFrame as ToDataFrameTrait, ToDataFrameVec,
    };
}

#[cfg(test)]
mod tests {
    use crate::ToDataFrame;

    #[derive(ToDataFrame)]
    struct SelfCrateRow {
        id: u32,
        label: String,
    }

    #[test]
    fn derive_uses_facade_runtime_inside_facade_crate() -> polars::prelude::PolarsResult<()> {
        use crate::dataframe::{ToDataFrame as _, ToDataFrameVec as _};

        let row = SelfCrateRow {
            id: 1,
            label: "facade".to_owned(),
        };
        let single = row.to_dataframe()?;
        assert_eq!(single.shape(), (1, 2));

        let rows = [
            row,
            SelfCrateRow {
                id: 2,
                label: "self".to_owned(),
            },
        ];
        let batch = rows.as_slice().to_dataframe()?;
        assert_eq!(batch.shape(), (2, 2));

        Ok(())
    }
}
