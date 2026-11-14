use std::fmt::Display;

/// A type with a fixed set of values, each of which has a name: an enum
/// whose variants hold no data.
///
/// This is what [`EnumBuilder`](super::builders::EnumBuilder) needs of a
/// type to store it in an `Enum` column. `#[derive(ToDataFrame)]` on such an
/// enum implements it. Implement it by hand for a type the derive does not
/// fit:
///
/// ```
/// use std::fmt::Display;
///
/// use df_derive_core::dataframe::builders::EnumBuilder;
/// use df_derive_core::dataframe::{Columnar, Variants};
///
/// /// A day of the week, counted from Monday.
/// struct Weekday(u8);
///
/// impl Variants for Weekday {
///     fn names() -> impl IntoIterator<Item = impl Display> {
///         ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
///     }
///
///     fn position(&self) -> usize {
///         usize::from(self.0)
///     }
/// }
///
/// impl Columnar for Weekday {
///     type Builder = EnumBuilder<Self>;
/// }
/// ```
///
/// The trait is part of the `categorical` feature, which is off by default.
pub trait Variants: 'static {
    /// The name of every variant, as anything that can be written as text.
    /// No two variants have the same name.
    ///
    /// The names are the categories of the column, in this order, which is
    /// also the order the column sorts in. They are asked for once, the
    /// first time the type is stored.
    fn names() -> impl IntoIterator<Item = impl Display>;

    /// The position of this value's name among the [`names`](Variants::names).
    ///
    /// A position past the last name is an encoding error.
    fn position(&self) -> usize;
}
