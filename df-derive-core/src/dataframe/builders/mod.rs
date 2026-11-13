//! Column builders for the supported leaf types and wrappers.
//!
//! These are public so a field can name one directly, either through the
//! derive's field attributes or with `#[df_derive(with = "...")]`.
//!
//! The methods that append are `#[inline(always)]` on purpose. Builders nest,
//! and a column is only filled by one tight loop when every layer between the
//! row and the buffer is inlined into it.

#![allow(clippy::inline_always)]

mod bits;
#[cfg(feature = "decimal")]
mod decimal;
mod list;
mod map;
mod option;
mod pointer;
mod primitive;
mod temporal;
mod text;
mod tuple;

#[cfg(feature = "decimal")]
pub use decimal::DecimalBuilder;
pub use list::ListBuilder;
pub use map::MapBuilder;
pub use option::OptionBuilder;
pub use pointer::PtrBuilder;
pub use primitive::{BoolBuilder, Primitive, PrimitiveBuilder};
#[cfg(feature = "chrono")]
pub use temporal::{DateBuilder, TimeOfDayBuilder};
pub use temporal::{Ms, Ns, TemporalBuilder, Unit, Us};
pub use text::{BinaryBuilder, DisplayBuilder, StrBuilder};
pub use tuple::UnitBuilder;
