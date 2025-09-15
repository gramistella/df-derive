//! df-derive – derive fast conversions from your Rust types to Polars `DataFrame`
//!
//! ## What this crate does
//!
//! Deriving `ToDataFrame` on your structs and tuple structs generates fast, allocation-conscious
//! code to:
//!
//! - Convert a single value to a `polars::prelude::DataFrame`
//! - Convert a slice of values via a columnar path (efficient batch conversion)
//! - Inspect the schema (column names and `DataType`s) at compile time via a generated method
//!
//! It supports nested structs (flattened with dot notation), `Option<T>`, `Vec<T>`, tuple structs,
//! and key domain types like `chrono::DateTime<Utc>` and `rust_decimal::Decimal`.
//!
//! ## Installation
//!
//! Add the macro crate and Polars. You will also need a trait defining the `to_dataframe` behavior
//! (you can use your own runtime crate/traits; see the override section below). For a minimal inline
//! trait you can copy, see the Quick start example.
//!
//! ```toml
//! [dependencies]
//! df-derive = "0.1"
//! polars = { version = "0.50", features = ["timezones", "dtype-decimal"] }
//!
//! # If you use these types in your models
//! chrono = { version = "0.4", features = ["serde"] }
//! rust_decimal = { version = "1.36", features = ["serde"] }
//! ```
//!
//! ## Quick start
//!
//! Copy-paste runnable example without any external runtime traits. This is a complete working
//! example that you can run with `cargo run --example quickstart`. In your own project, place the
//! `dataframe` traits wherever you like and point the derive macro to them (see
//! "Crate path override").
//!
//! ```rust
//! use df_derive::ToDataFrame;
//!
//! mod dataframe {
//!     use polars::prelude::{DataFrame, DataType, PolarsResult};
//!
//!     pub trait ToDataFrame {
//!         fn to_dataframe(&self) -> PolarsResult<DataFrame>;
//!         fn empty_dataframe() -> PolarsResult<DataFrame>;
//!         fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
//!     }
//!
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
//!     }
//! }
//!
//! #[derive(ToDataFrame)]
//! #[df_derive(trait = "crate::dataframe::ToDataFrame")] // Columnar path auto-infers
//! struct Trade { symbol: String, price: f64, size: u64 }
//!
//! fn main() -> polars::prelude::PolarsResult<()> {
//!     let t = Trade { symbol: "AAPL".into(), price: 187.23, size: 100 };
//!     let df_single = <Trade as crate::dataframe::ToDataFrame>::to_dataframe(&t)?;
//!     println!("{}", df_single);
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - **Nested structs (flattening)**: fields of nested structs appear as `outer.inner` columns
//! - **Vec of primitives and structs**: becomes Polars `List` columns; `Vec<Nested>` becomes
//!   multiple `outer.subfield` list columns
//! - **`Option<T>`**: null-aware materialization for both scalars and lists
//! - **Tuple structs**: supported; columns are named `field_0`, `field_1`, ...
//! - **Empty structs**: produce `(1, 0)` for instances and `(0, 0)` for empty frames
//! - **Schema discovery**: `T::schema() -> Vec<(&'static str, DataType)>`
//! - **Columnar batch conversion**: `[T]` via a generated `Columnar` implementation
//!
//! ### Attribute helpers
//!
//! Use `#[df_derive(as_string)]` to stringify values during conversion. This is particularly useful
//! for enums:
//!
//! ```rust
//! use df_derive::ToDataFrame;
//!
//! // Minimal runtime traits used by the derive macro
//! mod dataframe {
//!     use polars::prelude::{DataFrame, DataType, PolarsResult};
//!     pub trait ToDataFrame {
//!         fn to_dataframe(&self) -> PolarsResult<DataFrame>;
//!         fn empty_dataframe() -> PolarsResult<DataFrame>;
//!         fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
//!     }
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
//!     }
//! }
//!
//! #[derive(Clone, Debug, PartialEq)]
//! enum Status { Active, Inactive }
//!
//! impl std::fmt::Display for Status {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         match self {
//!             Status::Active => write!(f, "Active"),
//!             Status::Inactive => write!(f, "Inactive"),
//!         }
//!     }
//! }
//!
//! #[derive(ToDataFrame)]
//! #[df_derive(trait = "crate::dataframe::ToDataFrame")]
//! struct WithEnums {
//!     #[df_derive(as_string)]
//!     status: Status,
//!     #[df_derive(as_string)]
//!     opt_status: Option<Status>,
//!     #[df_derive(as_string)]
//!     statuses: Vec<Status>,
//! }
//!
//! fn main() {}
//! ```
//!
//! Columns will use `DataType::String` (or `List<String>` for `Vec<_>`), and values are produced via
//! `ToString`.
//!
//! ## Supported types
//!
//! - **Primitives**: `String`, `bool`, integer types (`i8/i16/i32/i64/isize`, `u8/u16/u32/u64/usize`),
//!   `f32`, `f64`
//! - **Time**: `chrono::DateTime<Utc>` → materialized as `Datetime(Milliseconds, None)`
//! - **Decimal**: `rust_decimal::Decimal` → `Decimal(38, 10)`
//! - **Wrappers**: `Option<T>`, `Vec<T>` in any nesting order
//! - **Custom structs**: any other struct deriving `ToDataFrame` (supports nesting and `Vec<Nested>`,
//!   yielding prefixed list columns)
//! - **Tuple structs**: unnamed fields are emitted as `field_{index}`
//!
//! ## Column naming
//!
//! - Named struct fields: `field_name`
//! - Nested structs: `outer.inner` (recursively)
//! - Vec of custom structs: `vec_field.subfield` (list dtype)
//! - Tuple structs: `field_0`, `field_1`, ...
//!
//! ## Generated API
//!
//! For every `#[derive(ToDataFrame)]` type `T` the macro generates implementations of two traits
//! (paths configurable via `#[df_derive(...)]`):
//!
//! - `ToDataFrame` for `T`:
//!   - `fn to_dataframe(&self) -> PolarsResult<DataFrame>`
//!   - `fn empty_dataframe() -> PolarsResult<DataFrame>`
//!   - `fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>`
//! - `Columnar` for `T`:
//!   - `fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>`
//!
//! Empty-struct behavior:
//!
//! - `to_dataframe(&self)` produces a single-row `DataFrame` with zero columns
//! - `empty_dataframe()` produces a `(0, 0)` `DataFrame`
//! - `columnar_to_dataframe(&[T])` produces a zero-column `DataFrame` with `items.len()` rows
//!
//! ## Examples
//!
//! This crate includes several runnable examples in the `examples/` directory:
//!
//! - `quickstart` — Basic usage with single and batch `DataFrame` conversion
//! - `nested` — Nested structs with dot notation column naming
//! - `vec_custom` — Vec of custom structs creating List columns
//! - `tuple` — Tuple structs with `field_0`, `field_1` naming
//! - `datetime_decimal` — `DateTime` and `Decimal` type support
//! - `as_string` — `#[df_derive(as_string)]` attribute for enum conversion
//!
//! ## Limitations and guidance
//!
//! - **Unsupported container types**: maps/sets like `HashMap<_, _>` are not supported
//! - **Enums**: derive on enums is not supported; use `#[df_derive(as_string)]` on enum fields
//! - **Generics**: generic structs are not supported by the derive (see `tests/fail`)
//! - **All nested types must also derive**: if you nest a struct, it must also derive `ToDataFrame`
//!
//! ## Performance notes
//!
//! The derive implements an internal `Columnar` path used by runtimes to convert slices efficiently,
//! avoiding per-row `DataFrame` builds. Criterion benches in `benches/` exercise wide, deep, and
//! nested-Vec shapes (100k+ rows), demonstrating consistent performance across shapes.
//!
//! ## Compatibility
//!
//! - **Rust edition**: 2024
//! - **Polars**: 0.50 (tested). Enable Polars features `timezones` and `dtype-decimal` if you use
//!   `DateTime<Utc>` or `Decimal`.
//!
//! ## License
//!
//! MIT. See `LICENSE`.
//!
//! ## Crate path override (about paft)
//!
//! By default, the macro resolves trait paths to a `dataframe` module under the `paft` ecosystem.
//! Concretely, it attempts to implement `paft::dataframe::ToDataFrame` and
//! `paft::dataframe::Columnar` (or `paft-core::dataframe::...`) if those crates are present. You can
//! override these paths for any runtime by annotating your type with `#[df_derive(...)]`:
//!
//! ```rust
//! use df_derive::ToDataFrame;
//!
//! // Define a local runtime with the expected traits
//! mod my_runtime { pub mod dataframe {
//!     use polars::prelude::{DataFrame, DataType, PolarsResult};
//!     pub trait ToDataFrame {
//!         fn to_dataframe(&self) -> PolarsResult<DataFrame>;
//!         fn empty_dataframe() -> PolarsResult<DataFrame>;
//!         fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
//!     }
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
//!     }
//! }}
//!
//! #[derive(ToDataFrame)]
//! #[df_derive(trait = "my_runtime::dataframe::ToDataFrame")] // Columnar inferred
//! struct MyType {}
//!
//! fn main() {}
//! ```
//!
//! If you need to override both explicitly:
//!
//! ```rust
//! use df_derive::ToDataFrame;
//!
//! // Define a local runtime with the expected traits
//! mod my_runtime { pub mod dataframe {
//!     use polars::prelude::{DataFrame, DataType, PolarsResult};
//!     pub trait ToDataFrame {
//!         fn to_dataframe(&self) -> PolarsResult<DataFrame>;
//!         fn empty_dataframe() -> PolarsResult<DataFrame>;
//!         fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
//!     }
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
//!     }
//! }}
//!
//! #[derive(ToDataFrame)]
//! #[df_derive(
//!     trait = "my_runtime::dataframe::ToDataFrame",
//!     columnar = "my_runtime::dataframe::Columnar",
//! )]
//! struct MyType {}
//!
//! fn main() {}
//! ```
#![warn(missing_docs)]
extern crate proc_macro;

mod codegen;
mod ir;
mod parser;
mod type_analysis;
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Derive `ToDataFrame` for structs and tuple structs to generate fast conversions to Polars.
///
/// What this macro generates (paths configurable via `#[df_derive(...)]`):
///
/// - An implementation of `ToDataFrame` for the annotated type `T` providing:
///   - `fn to_dataframe(&self) -> PolarsResult<DataFrame>`
///   - `fn empty_dataframe() -> PolarsResult<DataFrame>`
///   - `fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>`
/// - An implementation of `Columnar` for `T` providing
///   `fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>`
///
/// Supported shapes and types:
///
/// - Named and tuple structs (tuple fields are named `field_{index}`)
/// - Nested structs are flattened using dot notation (e.g., `outer.inner`)
/// - Wrappers `Option<T>` and `Vec<T>` in any nesting order, with `Vec<Struct>` producing multiple
///   list columns with a `vec_field.subfield` prefix
/// - Primitive types: `String`, `bool`, integer types, `f32`, `f64`
/// - `chrono::DateTime<Utc>` (materialized as `Datetime(Milliseconds, None)`)
/// - `rust_decimal::Decimal` (materialized as `Decimal(38, 10)`)
///
/// Attributes:
///
/// - Container-level: `#[df_derive(trait = "path::ToDataFrame")]` to set the `ToDataFrame` trait
///   path; the `Columnar` path is inferred by replacing the last path segment with `Columnar`.
///   Optionally, set both explicitly with
///   `#[df_derive(columnar = "path::Columnar")]`.
/// - Field-level: `#[df_derive(as_string)]` to stringify values (e.g., enums) during conversion,
///   resulting in `DataType::String` or `List<String>`.
///
/// Notes:
///
/// - Enums and generic structs are not supported for derive.
/// - All nested custom structs must also derive `ToDataFrame`.
/// - Empty structs: `to_dataframe` yields a single-row, zero-column `DataFrame`; the columnar path
///   yields a zero-column `DataFrame` with `items.len()` rows.
#[proc_macro_derive(ToDataFrame, attributes(df_derive))]
pub fn to_dataframe_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let ast = parse_macro_input!(input as DeriveInput);
    // Parse helper attribute configuration (trait paths)
    let default_df_mod = codegen::resolve_paft_crate_path();
    let mut to_df_trait_path_ts = quote! { #default_df_mod::ToDataFrame };
    let mut columnar_trait_path_ts = quote! { #default_df_mod::Columnar };

    for attr in &ast.attrs {
        if attr.path().is_ident("df_derive") {
            let parse_res = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("trait") {
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    let path: syn::Path = syn::parse_str(&lit.value())
                        .map_err(|e| meta.error(format!("invalid trait path: {e}")))?;
                    to_df_trait_path_ts = quote! { #path };

                    // Automatically infer the Columnar trait path by replacing the final segment
                    let mut columnar_path = path;
                    if let Some(last_segment) = columnar_path.segments.last_mut() {
                        last_segment.ident = syn::Ident::new("Columnar", last_segment.ident.span());
                    }
                    columnar_trait_path_ts = quote! { #columnar_path };
                    Ok(())
                } else if meta.path.is_ident("columnar") {
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    let path: syn::Path = syn::parse_str(&lit.value())
                        .map_err(|e| meta.error(format!("invalid columnar trait path: {e}")))?;
                    columnar_trait_path_ts = quote! { #path };
                    Ok(())
                } else {
                    Err(meta.error("unsupported key in #[df_derive(...)] attribute"))
                }
            });
            if let Err(err) = parse_res {
                return err.to_compile_error().into();
            }
        }
    }
    let config = codegen::MacroConfig {
        to_dataframe_trait_path: to_df_trait_path_ts,
        columnar_trait_path: columnar_trait_path_ts,
    };
    // Build the intermediate representation
    let ir = match parser::parse_to_ir(&ast) {
        Ok(ir) => ir,
        Err(e) => return e.to_compile_error().into(),
    };

    // Delegate to the codegen orchestrator
    let generated = codegen::generate_code(&ir, &config);
    TokenStream::from(generated)
}
