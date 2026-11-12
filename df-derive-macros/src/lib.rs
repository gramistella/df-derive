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

/// Derives `ToDataFrame` for a struct or tuple struct.
///
/// Each field becomes one or more columns. A field of any type that
/// implements the runtime's `Columnar` trait needs no attribute: numbers,
/// `bool`, strings, `chrono` dates and times, durations, decimals, other
/// derived structs (flattened with dot notation), tuples, and any nesting of
/// `Option`, `Vec`, `Box`, `Rc`, `Arc`, `&` and `Cow` around them.
///
/// # Container attributes
///
/// - `#[df_derive(crate = "path")]`: the crate that re-exports the runtime as
///   `path::dataframe`. Defaults to `::df_derive`.
///
/// # Field attributes
///
/// - `skip`: leave the field out.
/// - `flatten` / `flatten(prefix = "p")`: splice a nested struct's columns in
///   without the field-name prefix, or under an explicit prefix.
/// - `as_str`: borrow the value through `AsRef<str>` into a string column.
/// - `as_string`: format the value with `Display` into a string column.
/// - `as_binary`: borrow the value through `AsRef<[u8]>` into a binary column.
/// - `decimal(precision = P, scale = S)`: store a `Decimal128Encode` value as
///   `Decimal(P, S)`.
/// - `time_unit = "ms" | "us" | "ns"`: choose the unit of a datetime or
///   duration column.
/// - `with = "Builder"`: encode the field with a custom `ColumnBuilder`.
///
/// The encoding attributes apply to the innermost value, through any
/// `Option`, `Vec` and pointer layers written in the field's type.
#[proc_macro_derive(ToDataFrame, attributes(df_derive))]
pub fn derive_to_dataframe(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
