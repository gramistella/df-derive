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
//! and key domain types like `chrono::DateTime<Utc>`, `chrono::NaiveDateTime`, and decimal backend
//! paths named `Decimal`.
//!
//! ## Installation
//!
//! Add the macro crate, Polars, and `polars-arrow`. You will also need a trait defining the
//! `to_dataframe` behavior (you can use your own runtime crate/traits; see the override section
//! below). For a minimal inline trait you can copy, see the Quick start example.
//!
//! ```toml
//! [dependencies]
//! df-derive = "0.3.0"
//! polars = { version = "0.53", features = ["timezones", "dtype-date", "dtype-time", "dtype-duration", "dtype-decimal"] }
//! polars-arrow = "0.53"
//!
//! # If you use these types in your models
//! chrono = { version = "0.4", features = ["serde"] }
//! rust_decimal = { version = "1.36", features = ["serde"] }
//! ```
//!
//! `df-derive` requires `polars-arrow` as a direct dependency (it is compiled transitively via
//! `polars` but not re-exported through `polars::prelude`). Add `polars-arrow = "0.53"` to your
//! `Cargo.toml` alongside `polars`.
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
//!         fn schema() -> PolarsResult<Vec<(String, DataType)>>;
//!     }
//!
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
//!             let refs: Vec<&Self> = items.iter().collect();
//!             Self::columnar_from_refs(&refs)
//!         }
//!         fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
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
//! - **Schema discovery**: `T::schema() -> Vec<(String, DataType)>`
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
//!         fn schema() -> PolarsResult<Vec<(String, DataType)>>;
//!     }
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
//!             let refs: Vec<&Self> = items.iter().collect();
//!             Self::columnar_from_refs(&refs)
//!         }
//!         fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
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
//! - **Primitives**: `String`, `bool`, integer types
//!   (`i8/i16/i32/i64/i128/isize`, `u8/u16/u32/u64/u128/usize`), `f32`, `f64`
//! - **Time**: `chrono::DateTime<Utc>` and `chrono::NaiveDateTime` → materialized as
//!   `Datetime(Milliseconds, None)` by default; override per-field with
//!   `#[df_derive(time_unit = "ms"|"us"|"ns")]`
//! - **Date / time-of-day**: `chrono::NaiveDate` → `Date` (i32 days since 1970-01-01; requires
//!   Polars `dtype-date`), `chrono::NaiveTime` → `Time` (i64 ns since midnight; requires Polars
//!   `dtype-time`). Both have fixed encodings; `time_unit` is not accepted on either.
//! - **Duration**: `std::time::Duration`, `core::time::Duration`, and `chrono::Duration`
//!   (alias for `chrono::TimeDelta`) → `Duration(Nanoseconds)` by default (requires Polars
//!   `dtype-duration`); override per-field with `#[df_derive(time_unit = "ms"|"us"|"ns")]`.
//!   Bare `Duration` (no qualifier) is rejected as ambiguous — use `std::time::Duration`,
//!   `core::time::Duration`, or `chrono::Duration` to disambiguate.
//! - **Decimal**: any type path whose last segment is `Decimal` (for example
//!   `rust_decimal::Decimal` or a facade such as `paft_decimal::Decimal`) → `Decimal(38, 10)`
//!   by default. This implicit detection is syntax-based because proc macros cannot resolve
//!   type aliases. For differently named decimal backends, use
//!   `#[df_derive(decimal(precision = N, scale = N))]` and implement `Decimal128Encode`.
//! - **Binary blobs**: opt-in per field with `#[df_derive(as_binary)]` over a `Vec<u8>` or
//!   `Cow<'_, [u8]>` shape; the default for `Vec<u8>` (no attribute) remains `List(UInt8)`, while
//!   unannotated `Cow<'_, [u8]>` is rejected. See the field-level attribute list below for accepted
//!   shapes (`Vec<u8>`, `Option<Vec<u8>>`, `Vec<Vec<u8>>`, `Vec<Option<Vec<u8>>>`,
//!   `Option<Vec<Vec<u8>>>`, and the same scalar/list shapes over `Cow<'_, [u8]>`).
//! - **Wrappers**: `Option<T>`, `Vec<T>` in any nesting order
//! - **Smart pointers**: `Box<T>`, `Rc<T>`, `Arc<T>`, and `Cow<'_, T>` with sized inner peel
//!   transparently — they have no semantic effect on the column shape. `Option<Box<i32>>` resolves
//!   to the same `Int32` schema as `Option<i32>`; `Vec<Arc<String>>` to `List<String>`;
//!   `Box<Vec<f64>>` to `List<Float64>`; `Cow<'static, NaiveDate>` to `Date`. `Cow<'_, str>` is a
//!   borrowed string leaf by default. `Cow<'_, [u8]>` is supported with `#[df_derive(as_binary)]`;
//!   other `Cow<'_, [T]>` slice forms are rejected — use `Vec<T>` for list columns.
//! - **Custom structs**: any other struct deriving `ToDataFrame` (supports nesting and `Vec<Nested>`,
//!   yielding prefixed list columns)
//! - **Tuple structs**: unnamed fields are emitted as `field_{index}`
//! - **Tuple-typed fields**: `(A, B)`, `(A, B, C)`, `Option<(A, B)>`, `Vec<(A, B)>`,
//!   `Vec<Option<(A, B)>>`, `Option<Vec<(A, B)>>`, and unwrapped nested tuples
//!   `((A, B), C)` flatten to one column per element with `<field>.field_<i>` names.
//!   Nested tuples inside an outer `Option` or `Vec` are rejected; hoist the inner tuple
//!   into a named struct. The unit type `()` is rejected (zero-column fields would break
//!   the parser's invariant). Field-level attributes
//!   (`as_str`, `as_string`, `as_binary`, `decimal`, `time_unit`) do not apply to tuple
//!   fields — hoist into a named struct if per-element attributes are needed.
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
//!   - `fn schema() -> PolarsResult<Vec<(String, DataType)>>`
//! - `Columnar` for `T`:
//!   - `fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>` — borrowed entry point
//!     used by parent bulk emitters to avoid per-row clones; `columnar_to_dataframe` is provided
//!     by the trait's default and delegates to this method.
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
//! - `generics` — Generic struct support with type parameters
//! - `nested_options` — `Option<Option<Struct>>` field handling
//! - `deep_vec` — Deep `Vec<Vec<Vec<T>>>` nesting
//!
//! ## Limitations and guidance
//!
//! - **Unsupported container types**: maps/sets like `HashMap<_, _>` are not supported
//! - **Enums**: derive on enums is not supported; use `#[df_derive(as_string)]` on enum fields
//! - **Generics**: generic structs are supported. The macro injects `ToDataFrame + Columnar`
//!   bounds on every type parameter, plus `Decimal128Encode` for generic parameters explicitly
//!   annotated with `decimal(...)`. Use `()` as a payload type to contribute zero columns.
//! - **All nested types must also derive**: if you nest a struct, it must also derive `ToDataFrame`
//!
//! ## Performance notes
//!
//! The derive implements an internal `Columnar` path used by runtimes to convert slices efficiently,
//! avoiding per-row `DataFrame` builds. Criterion benches in `benches/` exercise wide, deep, and
//! nested-Vec shapes (100k+ rows), demonstrating consistent performance across shapes.
//!
//! ## Architecture (encoder IR)
//!
//! Code generation flows through a compositional encoder IR rather than per-shape bespoke
//! emitters. Each field's parsed type is normalized into a wrapper stack
//! (`Option`/`Vec` layers) sitting above a base type, then folded into an encoder that
//! emits three slots: top-of-function declarations, per-row push tokens, and a
//! finishing block that yields the field's `Series` (or, for nested structs, multiple
//! Series — one per inner schema column).
//!
//! Each leaf advertises one of two kinds. *Per-element-push* leaves (numerics, strings,
//! decimals, dates) consume one value at a time inside the columnar populator's per-row
//! loop. *Collect-then-bulk* leaves (nested user structs, generic parameters) gather
//! references to inner values across all rows and dispatch a single
//! `<T as Columnar>::columnar_from_refs(&refs)` call, so nested derives compose without
//! per-row trait indirection. Both kinds are needed because the trade-off flips at the
//! base type: primitives lose to call-frame overhead from a per-row trait call, while
//! nested structs gain by amortizing the inner derive's setup once.
//!
//! Consecutive `Vec` layers are fused. For a `Vec<Vec<…<Vec<T>>>>` field the encoder
//! emits one flat values buffer at the deepest layer plus one pair of offsets per `Vec`
//! layer, all stacked into nested `LargeListArray`s in a single block. The bulk-fusion
//! invariant — that `vec(...)` collapses across consecutive layers rather than
//! emitting one populator per layer — is what makes deep-list shapes O(total leaf
//! count) instead of O(layer count × leaf count).
//!
//! `unsafe` is localized: the only call to
//! `Series::from_chunks_and_dtype_unchecked` lives in a free helper named
//! `__df_derive_assemble_list_series_unchecked`, hidden inside the per-derive
//! anonymous-`const` scope. Since no impl method on `Self` contains `unsafe`,
//! `clippy::unsafe_derive_deserialize` does not fire on user types that combine
//! `#[derive(ToDataFrame, Deserialize)]`.
//!
//! Several shapes use direct polars-arrow array construction in place of
//! `Series::new` or typed builders. Bypassing the typed-builder layer for
//! `Vec<numeric>`, `Vec<Option<numeric>>`, and the nested-`Vec<Struct>` family
//! consistently wins large multiples on bench shapes; the encoder routes those
//! cases to the direct-array path automatically.
//!
//! Every primitive shape flows through the encoder IR — bare leaves,
//! arbitrary `Option<…<Option<T>>>` stacks, every vec-bearing wrapper stack
//! including `[Option, Vec, ...]` over numeric / `String` / `Decimal` /
//! `DateTime` primitives, and `isize` / `usize` (widened to `i64` / `u64`
//! at the codegen boundary).
//!
//! ## Compatibility
//!
//! - **Rust edition**: 2024
//! - **Polars**: 0.53 (tested). Enable Polars features `timezones` for timezone-aware
//!   `DateTime<Utc>`, `dtype-date` for `NaiveDate`, `dtype-time` for `NaiveTime`,
//!   `dtype-duration` for duration columns, `dtype-i128` / `dtype-u128` for 128-bit
//!   integer columns, and `dtype-decimal` for `Decimal`.
//! - **polars-arrow**: 0.53 (direct dependency required by generated code).
//!
//! ## License
//!
//! MIT. See `LICENSE`.
//!
//! ## Crate path override (about paft)
//!
//! By default, the macro resolves trait paths to a `dataframe` module under the `paft` ecosystem.
//! Concretely, it attempts the `paft` facade first (`paft::dataframe::...`), then
//! `paft-utils` (`paft_utils::dataframe::...`), then a local `crate::core::dataframe`
//! fallback for projects that keep dataframe traits there without using paft. You can
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
//!         fn schema() -> PolarsResult<Vec<(String, DataType)>>;
//!     }
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
//!             let refs: Vec<&Self> = items.iter().collect();
//!             Self::columnar_from_refs(&refs)
//!         }
//!         fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
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
//!         fn schema() -> PolarsResult<Vec<(String, DataType)>>;
//!     }
//!     pub trait Columnar: Sized {
//!         fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
//!             let refs: Vec<&Self> = items.iter().collect();
//!             Self::columnar_from_refs(&refs)
//!         }
//!         fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
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

/// Parse a `key = "path::Trait"` attribute value into a `syn::Path`, with a
/// uniform error message of the form `"invalid {label} path: {e}"`. Callers
/// pass the full noun phrase (e.g., `"trait"`, `"columnar trait"`,
/// `"decimal128_encode trait"`) so the existing user-facing strings are
/// preserved verbatim.
fn parse_trait_path_attr(
    meta: &syn::meta::ParseNestedMeta<'_>,
    label: &str,
) -> syn::Result<syn::Path> {
    let lit: syn::LitStr = meta.value()?.parse()?;
    syn::parse_str(&lit.value()).map_err(|e| meta.error(format!("invalid {label} path: {e}")))
}

/// Clone `path` and replace the last segment's identifier with `name`,
/// preserving the original span. Used to derive sibling trait paths
/// (`Columnar`, `Decimal128Encode`) from a user-supplied `ToDataFrame` path.
fn rebase_last_segment(path: &syn::Path, name: &str) -> syn::Path {
    let mut new_path = path.clone();
    if let Some(last_segment) = new_path.segments.last_mut() {
        last_segment.ident = syn::Ident::new(name, last_segment.ident.span());
    }
    new_path
}

/// Derive `ToDataFrame` for structs and tuple structs to generate fast conversions to Polars.
///
/// What this macro generates (paths configurable via `#[df_derive(...)]`):
///
/// - An implementation of `ToDataFrame` for the annotated type `T` providing:
///   - `fn to_dataframe(&self) -> PolarsResult<DataFrame>`
///   - `fn empty_dataframe() -> PolarsResult<DataFrame>`
///   - `fn schema() -> PolarsResult<Vec<(String, DataType)>>`
/// - An implementation of `Columnar` for `T` providing
///   `fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>`
///   (`columnar_to_dataframe` uses the trait default and delegates to this method)
///
/// Supported shapes and types:
///
/// - Named and tuple structs (tuple fields are named `field_{index}`)
/// - Nested structs are flattened using dot notation (e.g., `outer.inner`)
/// - Wrappers `Option<T>` and `Vec<T>` in any nesting order, with `Vec<Struct>` producing multiple
///   list columns with a `vec_field.subfield` prefix
/// - Primitive types: `String`, `bool`, integer types including `i128`/`u128`, `f32`, `f64`
/// - `chrono::DateTime<Utc>` and `chrono::NaiveDateTime` (default:
///   `Datetime(Milliseconds, None)`; override with `#[df_derive(time_unit = "ms"|"us"|"ns")]`)
/// - `chrono::NaiveDate` (`Date`, i32 days since 1970-01-01) and `chrono::NaiveTime`
///   (`Time`, i64 ns since midnight); both have fixed encodings, no unit override.
/// - `std::time::Duration`, `core::time::Duration`, and `chrono::Duration` (alias for
///   `chrono::TimeDelta`) → `Duration(Nanoseconds)` by default; override with
///   `#[df_derive(time_unit = "ms"|"us"|"ns")]`. Bare `Duration` is ambiguous and rejected.
/// - Decimal backends written with a final `Decimal` path segment, such as
///   `rust_decimal::Decimal` or `paft_decimal::Decimal` (default: `Decimal(38, 10)`;
///   override with `#[df_derive(decimal(precision = N, scale = N))]`). This is a
///   syntax-level heuristic, not type resolution; differently named backends opt in with
///   explicit `decimal(...)` and a `Decimal128Encode` impl.
///
/// Attributes:
///
/// - Container-level: `#[df_derive(trait = "path::ToDataFrame")]` to set the `ToDataFrame` trait
///   path; the `Columnar` and `Decimal128Encode` paths are inferred by replacing the last
///   path segment with `Columnar` / `Decimal128Encode`. Optionally, set them explicitly with
///   `#[df_derive(columnar = "path::Columnar")]` and
///   `#[df_derive(decimal128_encode = "path::Decimal128Encode")]`. The latter is the dispatch
///   point for `rust_decimal::Decimal` / `bigdecimal::BigDecimal` / other decimal backends —
///   see "Custom decimal backends" in the README for the trait contract.
/// - Field-level: `#[df_derive(as_string)]` to stringify values via `Display` (e.g., enums) during
///   conversion, resulting in `DataType::String` or `List<String>`. Allocates a `String` per row.
/// - Field-level: `#[df_derive(as_str)]` to borrow `&str` via `AsRef<str>` for the duration of the
///   conversion. Same column type as `as_string` but avoids the per-row allocation. The two
///   attributes are mutually exclusive on a given field.
/// - Field-level: `#[df_derive(as_binary)]` to route a `Vec<u8>` or `Cow<'_, [u8]>` field through a
///   Polars `Binary` column instead of the default `List(UInt8)` for `Vec<u8>`. Accepted shapes:
///   `Vec<u8>`, `Option<Vec<u8>>`, `Vec<Vec<u8>>`, `Vec<Option<Vec<u8>>>`,
///   `Option<Vec<Vec<u8>>>`, and the same scalar/list shapes over `Cow<'_, [u8]>` — bare `u8`,
///   `Option<u8>`, `Vec<Option<u8>>` (`BinaryView` cannot carry per-byte nulls), and non-`u8`
///   leaves are rejected at parse time. Mutually exclusive with `as_str`,
///   `as_string`, `decimal(...)`, and `time_unit = "..."`.
/// - Field-level: `#[df_derive(decimal(precision = N, scale = N))]` to choose the
///   `Decimal(precision, scale)` dtype for a path named `Decimal` or to explicitly opt a
///   custom/generic decimal backend into `Decimal128Encode` dispatch. Polars requires
///   `1 <= precision <= 38`; `scale` may not exceed `precision`.
/// - Field-level: `#[df_derive(time_unit = "ms"|"us"|"ns")]` to choose the
///   `Datetime(unit, None)` / `Duration(unit)` dtype for a temporal field. Accepted bases are
///   `chrono::DateTime<Utc>`, `chrono::NaiveDateTime`, `std::time::Duration`,
///   `core::time::Duration`, and `chrono::Duration`. The chrono / std call used to derive the
///   i64 matches the chosen unit, so values are not silently truncated. `time_unit = "ns"` on
///   `DateTime<Utc>` or `NaiveDateTime` is fallible on dates outside chrono's supported
///   nanosecond range (~1677–2262); `time_unit = "ns"`/`"us"` on `chrono::Duration` is fallible
///   when the duration overflows i64 in the chosen unit; on `std::time::Duration` every unit is
///   fallible (the value type is `u128`). All failures surface as `PolarsError::ComputeError`
///   rather than silently corrupting data. `time_unit` is rejected on `chrono::NaiveDate` and
///   `chrono::NaiveTime` (both have fixed encodings).
/// - The `decimal(...)` attribute can only be applied to decimal backend candidates: type paths
///   named `Decimal`, custom struct types, or generic type parameters that implement
///   `Decimal128Encode`. It cannot be combined with `as_str`/`as_string`/`time_unit` on the same
///   field. The `time_unit = "..."` attribute is also mutually exclusive with
///   `as_str`/`as_string`.
///
/// Notes:
///
/// - Enums are not supported for derive.
/// - Generic structs are supported; the macro injects `ToDataFrame + Columnar`
///   bounds on every type parameter, plus `Decimal128Encode` for generic parameters explicitly
///   annotated with `decimal(...)`. The unit type `()` is a valid payload (zero columns).
/// - All nested custom structs must also derive `ToDataFrame`.
/// - Empty structs: `to_dataframe` yields a single-row, zero-column `DataFrame`; the columnar path
///   yields a zero-column `DataFrame` with `items.len()` rows.
#[proc_macro_derive(ToDataFrame, attributes(df_derive))]
pub fn to_dataframe_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let ast = parse_macro_input!(input as DeriveInput);
    // Parse helper attribute configuration (trait paths)
    let default_df_mod = codegen::resolve_paft_crate_path();
    let mut to_df_trait_path: Option<syn::Path> = None;
    let mut columnar_trait_path: Option<syn::Path> = None;
    let mut decimal128_encode_trait_path: Option<syn::Path> = None;

    for attr in &ast.attrs {
        if attr.path().is_ident("df_derive") {
            let parse_res = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("trait") {
                    let path = parse_trait_path_attr(&meta, "trait")?;
                    to_df_trait_path = Some(path);
                    Ok(())
                } else if meta.path.is_ident("columnar") {
                    let path = parse_trait_path_attr(&meta, "columnar trait")?;
                    columnar_trait_path = Some(path);
                    Ok(())
                } else if meta.path.is_ident("decimal128_encode") {
                    let path = parse_trait_path_attr(&meta, "decimal128_encode trait")?;
                    decimal128_encode_trait_path = Some(path);
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

    let to_df_trait_path_ts = to_df_trait_path.as_ref().map_or_else(
        || quote! { #default_df_mod::ToDataFrame },
        |path| quote! { #path },
    );
    let columnar_trait_path_ts = match (&columnar_trait_path, &to_df_trait_path) {
        (Some(path), _) => quote! { #path },
        (None, Some(path)) => {
            let columnar_path = rebase_last_segment(path, "Columnar");
            quote! { #columnar_path }
        }
        (None, None) => quote! { #default_df_mod::Columnar },
    };
    let decimal128_encode_trait_path_ts = match (&decimal128_encode_trait_path, &to_df_trait_path) {
        (Some(path), _) => quote! { #path },
        (None, Some(path)) => {
            let decimal_path = rebase_last_segment(path, "Decimal128Encode");
            quote! { #decimal_path }
        }
        (None, None) => quote! { #default_df_mod::Decimal128Encode },
    };

    let config = codegen::MacroConfig {
        to_dataframe_trait_path: to_df_trait_path_ts,
        columnar_trait_path: columnar_trait_path_ts,
        decimal128_encode_trait_path: decimal128_encode_trait_path_ts,
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
