//! Derive fast conversions from Rust structs into Polars `DataFrame`s.
//!
//! ```toml
//! [dependencies]
//! df-derive = "0.6"
//! polars = "0.55"
//! ```
//!
//! ```ignore
//! use df_derive::prelude::*;
//!
//! #[derive(ToDataFrame)]
//! struct Trade {
//!     symbol: String,
//!     price: f64,
//!     size: Option<u64>,
//! }
//!
//! let df = trades.as_slice().to_dataframe()?;
//! ```
//!
//! The derive targets the runtime in [`dataframe`]. Any field whose type
//! implements [`dataframe::Columnar`] works without an attribute; see the
//! derive macro's documentation for the field and container attributes, and
//! the repository README for the full list of supported types.
//!
//! A crate that re-exports this runtime under its own path tells the derive
//! where to find it with `#[df_derive(crate = "::that_crate")]`.

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
    pub use crate::dataframe::{
        ColumnBuilder, Columnar, Decimal128Encode, ToDataFrame, ToDataFrame as ToDataFrameTrait,
        ToDataFrameVec,
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
