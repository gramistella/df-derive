//! User-facing facade for deriving Polars `DataFrame` conversions.
//!
//! Most users should depend on this crate, import the prelude, and derive
//! `ToDataFrame` without any runtime-path attributes:
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
