//! Proc-macro implementation crate for `df-derive`.
//!
//! Most users should depend on the `df-derive` facade, which re-exports
//! this macro and the default runtime traits from `df-derive-core`. Depend
//! on `df-derive-macros` directly when you want to target `paft`,
//! `df-derive-core`, or a custom runtime without the facade.
//!
//! Explicit `#[df_derive(trait = "...")]` selects a custom runtime path.
//! Explicit paths to the built-in `df_derive::dataframe::ToDataFrame` or
//! `df_derive_core::dataframe::ToDataFrame` runtimes are treated as the
//! default runtime and still use the runtime's hidden dependency re-exports.
//! `columnar = "..."` may be provided alongside `trait = "..."`, and
//! `decimal128_encode = "..."` may override decimal dispatch. Built-in
//! dataframe runtime paths cannot be mixed with custom `columnar` paths.
//! Without runtime overrides, discovery tries `df-derive`, `df-derive-core`,
//! `paft-utils`, `paft`, then the `crate::core::dataframe` fallback.
#![warn(missing_docs)]
extern crate proc_macro;

mod attrs;
mod codegen;
mod diagnostics;
mod ir;
mod lower;
mod parser;
mod type_analysis;
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

fn runtime_trait_path(dataframe_mod: &proc_macro2::TokenStream, trait_name: &str) -> syn::Path {
    let trait_ident = syn::Ident::new(trait_name, proc_macro2::Span::call_site());
    syn::parse2(quote! { #dataframe_mod::#trait_ident })
        .expect("default dataframe runtime trait path should parse")
}

fn build_macro_config(ast: &DeriveInput) -> syn::Result<codegen::MacroConfig> {
    let default_df_mod = codegen::resolve_default_dataframe_mod();
    let attrs = attrs::container::parse_container_attrs(ast)?;

    let uses_default_dataframe_runtime = attrs.to_dataframe.is_none() && attrs.columnar.is_none();
    let explicit_default_dataframe_mod = attrs::container::explicit_builtin_default_dataframe_mod(
        attrs.to_dataframe.as_ref(),
        attrs.columnar.as_ref(),
    );
    let to_dataframe = attrs.to_dataframe.as_ref().map_or_else(
        || runtime_trait_path(&default_df_mod, "ToDataFrame"),
        |override_| override_.path.clone(),
    );
    let columnar = match (&attrs.columnar, &attrs.to_dataframe) {
        (Some(override_), _) => override_.path.clone(),
        (None, Some(override_)) => {
            attrs::container::rebase_last_segment(&override_.path, "Columnar")
        }
        (None, None) => runtime_trait_path(&default_df_mod, "Columnar"),
    };
    let decimal128_encode = match (&attrs.decimal128_encode, &attrs.to_dataframe) {
        (Some(override_), _) => override_.path.clone(),
        (None, Some(override_)) => {
            attrs::container::rebase_last_segment(&override_.path, "Decimal128Encode")
        }
        (None, None) => runtime_trait_path(&default_df_mod, "Decimal128Encode"),
    };
    let external_paths = explicit_default_dataframe_mod.as_ref().map_or_else(
        || {
            if uses_default_dataframe_runtime {
                codegen::external_paths::default_runtime_paths(&default_df_mod)
            } else {
                codegen::external_paths::direct_dependency_paths()
            }
        },
        |dataframe_mod| {
            let dataframe_mod = quote! { #dataframe_mod };
            codegen::external_paths::default_runtime_paths(&dataframe_mod)
        },
    );

    Ok(codegen::MacroConfig {
        traits: codegen::RuntimeTraitPaths {
            to_dataframe,
            columnar,
            decimal128_encode,
        },
        external_paths,
    })
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
///   `fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>` and
///   `fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>`.
///   The direct slice method avoids the trait default's temporary ref-vector
///   allocation on top-level batch conversion; the borrowed method remains
///   available for nested and generic composition.
///
/// Supported shapes and types:
///
/// - Named and tuple structs (tuple fields are named `field_{index}`)
/// - Nested structs are flattened using dot notation (e.g., `outer.inner`)
/// - Wrappers `Option<T>` and `Vec<T>` in any nesting order, with `Vec<Struct>` producing multiple
///   list columns with a `vec_field.subfield` prefix
/// - Primitive types: `String`, `bool`, integer types including `i128`/`u128`,
///   `std::num::NonZero*` integer types, `f32`, `f64`
/// - `chrono::DateTime<Tz>` and `chrono::NaiveDateTime` (default:
///   `Datetime(Milliseconds, None)`; override with `#[df_derive(time_unit = "ms"|"us"|"ns")]`).
///   `DateTime<Tz>` stores the UTC instant; use `as_string` when the textual timezone/offset
///   matters.
/// - `chrono::NaiveDate` (`Date`, i32 days since 1970-01-01) and `chrono::NaiveTime`
///   (`Time`, i64 ns since midnight); both have fixed encodings, no unit override.
/// - `std::time::Duration`, `core::time::Duration`, and `chrono::Duration` (alias for
///   `chrono::TimeDelta`) → `Duration(Nanoseconds)` by default; override with
///   `#[df_derive(time_unit = "ms"|"us"|"ns")]`. Bare `Duration` is ambiguous and rejected.
/// - Decimal backends written as bare `Decimal` or `rust_decimal::Decimal`
///   (default: `Decimal(38, 10)`; override with
///   `#[df_derive(decimal(precision = N, scale = N))]`). Custom backends opt in
///   with explicit `decimal(...)` and a `Decimal128Encode` impl.
///
/// Attributes:
///
/// - Container-level: `#[df_derive(trait = "path::ToDataFrame")]` to set the `ToDataFrame` trait
///   path; the `Columnar` and `Decimal128Encode` paths are inferred by replacing the last
///   path segment with `Columnar` / `Decimal128Encode`. Optionally, set them explicitly with
///   `#[df_derive(columnar = "path::Columnar")]` and
///   `#[df_derive(decimal128_encode = "path::Decimal128Encode")]`. A `columnar` override
///   must be paired with `trait` to avoid mixed-runtime impls. `decimal128_encode` is the
///   dispatch point for `rust_decimal::Decimal` / `bigdecimal::BigDecimal` / other decimal
///   backends — see "Custom decimal backends" in the README for the trait contract. Explicit
///   paths to `df_derive::dataframe::ToDataFrame` or
///   `df_derive_core::dataframe::ToDataFrame` keep using the default runtime's hidden
///   dependency re-exports and cannot be paired with a custom `columnar` path;
///   other explicit trait paths are treated as custom runtimes.
/// - Field-level: `#[df_derive(skip)]` to omit a field from generated schema
///   and `DataFrame` output. Skipped fields are not type-analyzed, so this can
///   be used for caches, handles, source metadata, or other helper values that
///   should remain on the Rust struct but not become columns. Mutually
///   exclusive with conversion attributes.
/// - Field-level: `#[df_derive(as_string)]` to stringify values via `Display` (e.g., enums) during
///   conversion, resulting in `DataType::String` or `List<String>`. Generated encoders reuse a
///   `String` scratch buffer per field; the column builder still copies the formatted bytes.
/// - Field-level: `#[df_derive(as_str)]` to borrow `&str` via `AsRef<str>` for the duration of the
///   conversion. Same column type as `as_string` but avoids `Display` formatting and the
///   intermediate scratch buffer. The two attributes are mutually exclusive on a given field.
/// - Field-level: `#[df_derive(as_binary)]` to route a `Vec<u8>`, `&[u8]`, or
///   `Cow<'_, [u8]>` field through a Polars `Binary` column instead of the default
///   `List(UInt8)` for `Vec<u8>`. Accepted shapes:
///   `Vec<u8>`, `Option<Vec<u8>>`, `Vec<Vec<u8>>`, `Vec<Option<Vec<u8>>>`,
///   `Option<Vec<Vec<u8>>>`, and the same scalar/list shapes over `&[u8]` and `Cow<'_, [u8]>` —
///   bare `u8`, `Option<u8>`, `Vec<Option<u8>>` (`BinaryView` cannot carry per-byte nulls), and
///   non-`u8` leaves are rejected at parse time. Mutually exclusive with `as_str`,
///   `as_string`, `decimal(...)`, and `time_unit = "..."`.
/// - Field-level: `#[df_derive(decimal(precision = N, scale = N))]` to choose the
///   `Decimal(precision, scale)` dtype for a built-in decimal path or to explicitly opt a
///   custom/generic decimal backend into `Decimal128Encode` dispatch. Polars requires
///   `1 <= precision <= 38`; `scale` may not exceed `precision`.
/// - Field-level: `#[df_derive(time_unit = "ms"|"us"|"ns")]` to choose the
///   `Datetime(unit, None)` / `Duration(unit)` dtype for a temporal field. Accepted bases are
///   `chrono::DateTime<Tz>`, `chrono::NaiveDateTime`, `std::time::Duration`,
///   `core::time::Duration`, and `chrono::Duration`. The chrono / std call used to derive the
///   i64 matches the chosen unit, so values are not silently truncated. `time_unit = "ns"` on
///   `DateTime<Tz>` or `NaiveDateTime` is fallible on dates outside chrono's supported
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
/// - Generic structs are supported; the macro adds bounds only for the roles a
///   generic parameter actually plays (`ToDataFrame + Columnar` for nested
///   dataframe payloads, `AsRef<str>` for generic `as_str`, and
///   `Decimal128Encode` for generic decimal backends). The unit type `()` is a
///   valid generic payload (zero columns); direct `field: ()` fields are
///   rejected.
/// - All nested custom structs must also derive `ToDataFrame`.
/// - Obvious direct self-recursive nested fields using `Self`, the bare
///   deriving struct name, `self::Type`, or `crate::Type` are rejected after
///   transparent wrapper peeling, including `Box<T>`/`Option<Box<T>>` forms
///   and tuple fields containing the same.
/// - Empty structs: `to_dataframe` yields a single-row, zero-column `DataFrame`; the columnar path
///   yields a zero-column `DataFrame` with `items.len()` rows.
#[proc_macro_derive(ToDataFrame, attributes(df_derive))]
pub fn to_dataframe_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let ast = parse_macro_input!(input as DeriveInput);
    let config = match build_macro_config(&ast) {
        Ok(config) => config,
        Err(e) => return e.to_compile_error().into(),
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
