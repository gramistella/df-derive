//! Column builders for the supported leaf types and wrappers.
//!
//! These are public so a field can name one directly, either through the
//! derive's field attributes or with `#[df_derive(with = "...")]`.
//!
//! The hot `push` paths are `#[inline(always)]` on purpose: a row loop only
//! matches hand-written column code when every field push is inlined into it.

#![allow(clippy::inline_always)]

mod decimal;
mod list;
mod option;
mod pointer;
mod primitive;
mod temporal;
mod text;
mod tuple;

pub use decimal::DecimalBuilder;
pub use list::ListBuilder;
pub use option::OptionBuilder;
pub use pointer::PtrBuilder;
pub use primitive::{BoolBuilder, Primitive, PrimitiveBuilder};
#[cfg(feature = "chrono")]
pub use temporal::{DateBuilder, TimeOfDayBuilder};
pub use temporal::{Ms, Ns, TemporalBuilder, Unit, Us};
pub use text::{BinaryBuilder, DisplayBuilder, StrBuilder};
pub use tuple::UnitBuilder;
