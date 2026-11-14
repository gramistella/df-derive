//! Proc-macro implementation crate for `df-derive`.
//!
//! Depend on the `df-derive` facade instead; it re-exports this macro
//! together with the runtime the generated code targets.
//!
//! The derive does not analyse field types. For each field it names the
//! builder the runtime's `Columnar` trait selects for that type, so nested
//! structs, wrappers, aliases and generics are resolved by the compiler.
//! Only fields that carry an encoding attribute have their wrapper layers
//! (`Option`, `Vec`, pointers) read from the type's spelling.
#![warn(missing_docs)]

mod attrs;
mod expand;
mod mirror;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derives `ToDataFrame` for a struct or tuple struct, and makes an enum
/// whose variants hold no data a value that is one column.
///
/// Each field of a struct becomes one or more columns. A field of any type
/// that implements the runtime's `Columnar` trait needs no attribute:
/// numbers, `bool`, strings, `chrono` dates and times, durations, decimals,
/// other derived structs (flattened with dot notation), derived enums,
/// tuples, maps, and any nesting of `Option`, `Vec`, `Box`, `Rc`, `Arc`, `&`
/// and `Cow` around them.
///
/// An enum whose variants hold no data becomes an `Enum` column wherever it
/// is a field: the derive implements the runtime's `Variants` and `Columnar`
/// for it. The column's type lists the names of the variants in the order
/// they are declared in, and a row holds the position of its variant. A
/// variant is named by its identifier unless an attribute says otherwise.
/// Such an enum is a value, not a row, so it gets no `to_dataframe` of its
/// own. It needs the runtime's `categorical` feature. An enum whose variants
/// hold data is stored as text, with a field attribute, or spread over
/// columns with `via`.
///
/// A variant added to the enum changes the list, and with it the type of
/// the column. With `categorical` on the enum the column is a `Categorical`
/// one of the same names, whose type lists nothing. An enum that is
/// `#[non_exhaustive]` has to ask for that.
///
/// # Container attributes
///
/// - `crate = "path"`: the crate that re-exports the runtime as
///   `path::dataframe`. Defaults to `::df_derive`.
/// - `transparent`: a single-field wrapper encodes as its field, under the
///   name of whichever field holds the wrapper.
/// - `via = "Row"` / `try_via = "Row"`: encode through a row struct built
///   from a reference to each value, with `From<&T>` or `TryFrom<&T>`. Write
///   the lifetimes the row borrows from the value as `'_`. This is how an
///   enum whose variants hold data becomes several columns.
/// - `bound = "T: Trait, ..."`: extra where-clause predicates for the
///   generated impls.
/// - `as_str` / `as_string`, on an enum: name every variant by what the
///   enum's `AsRef<str>` gives for it, or by what its `Display` writes.
/// - `categorical`, on an enum: store it as a `Categorical` column of the
///   names of its variants, not as an `Enum` column that lists them.
///
/// # Field attributes
///
/// - `skip`: leave the field out.
/// - `rename = "name"`: use `name` in place of the field's own name, as the
///   column's name or as the prefix of a nested struct's columns.
/// - `flatten` / `flatten(prefix = "p")`: splice a nested struct's columns in
///   without the field-name prefix, or under an explicit prefix.
/// - `as_str`: borrow the value through `AsRef<str>` into a string column.
/// - `as_string`: format the value with `Display` into a string column.
/// - `as_binary`: borrow the value through `AsRef<[u8]>` into a binary column.
/// - `categorical`: store the value's text as a categorical column. The text
///   is read as `as_str` reads it, or with `as_string, categorical` as
///   `Display` writes it. It needs the runtime's `categorical` feature.
/// - `decimal(precision = P, scale = S)`: store a `Decimal128Encode` value as
///   `Decimal(P, S)`. It needs the runtime's `decimal` feature, which
///   `rust_decimal` turns on.
/// - `time_unit = "ms" | "us" | "ns"`: choose the unit of a datetime or
///   duration column. Without it a datetime is stored in microseconds and a
///   duration in nanoseconds.
/// - `with = "Builder"`: encode the value with a custom `ColumnBuilder`.
///
/// The encoding attributes apply to the innermost value, through any
/// `Option`, `Vec`, slice, array and pointer layers written in the field's
/// type. For a
/// `HashMap` or `BTreeMap` that is the map's value; its keys keep the
/// encoding of their own type.
///
/// # Variant attributes
///
/// - `rename = "name"`: use `name` in place of the variant's identifier as
///   its name in the column's type. It does not combine with `as_str` or
///   `as_string` on the enum.
#[proc_macro_derive(ToDataFrame, attributes(df_derive))]
pub fn derive_to_dataframe(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
