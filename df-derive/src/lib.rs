//! User-facing facade for deriving Polars `DataFrame` conversions.
//!
//! Most users should depend on this crate, import the prelude, and derive
//! `ToDataFrame` without any runtime-path attributes:
//!
//! ```toml
//! [dependencies]
//! df-derive = "0.3"
//! polars = "0.53"
//! ```
//!
//! The default facade hides the macro's `polars-arrow` implementation
//! dependency behind `df_derive::dataframe`; custom runtimes still need to
//! provide their own compatible direct dependencies.
//!
//! ```ignore
//! use df_derive::prelude::*;
//!
//! #[derive(ToDataFrame)]
//! struct Trade {
//!     symbol: String,
//!     price: f64,
//!     size: u64,
//! }
//! ```
//!
//! The derive macro targets [`dataframe`] by default, which is re-exported
//! from `df-derive-core`. Power users can depend on `df-derive-macros`
//! directly or use `#[df_derive(trait = "...")]`,
//! `#[df_derive(columnar = "...")]`, and
//! `#[df_derive(decimal128_encode = "...")]` to target a custom runtime.

// `polars` pulls a wide transitive dependency tree where multiple resolved
// versions are unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables; allow it here so linting stays
// focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

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
        Columnar, Decimal128Encode, ToDataFrame, ToDataFrame as ToDataFrameTrait, ToDataFrameVec,
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
