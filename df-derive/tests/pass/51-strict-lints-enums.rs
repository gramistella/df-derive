//! The derive's output for enums and categories under strict lints. See
//! `50-strict-lints.rs`.
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

use df_derive::ToDataFrame;

/// An enum stored as an `Enum` column.
#[derive(ToDataFrame, Debug, Clone, Copy)]
pub enum Side {
    /// Buy.
    Buy,
    /// Sell.
    #[df_derive(rename = "SELL")]
    Sell,
}

/// An enum that may grow, stored as categories.
#[derive(ToDataFrame, Debug, Clone, Copy)]
#[df_derive(categorical)]
#[non_exhaustive]
pub enum Venue {
    /// Lit.
    Lit,
    /// Dark.
    Dark,
}

/// A row that holds them.
#[derive(ToDataFrame, Debug, Clone)]
pub struct Order {
    /// The side.
    pub side: Side,
    /// The venue.
    pub venue: Option<Venue>,
    /// Text stored as categories.
    #[df_derive(categorical)]
    pub currency: String,
}

fn main() {}
