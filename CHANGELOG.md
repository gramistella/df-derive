# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Restructured the crate family into `df-derive` (facade),
  `df-derive-core` (shared runtime trait identity), and
  `df-derive-macros` (proc macro implementation). Normal users can now
  depend on `df-derive` and derive without a runtime-path override.
- Generated `Columnar` impls now override both `columnar_to_dataframe(&[Self])`
  and `columnar_from_refs(&[&Self])`, avoiding the top-level `Vec<&Self>`
  allocation while preserving borrowed nested composition.
- Default runtime discovery now checks `paft`, `paft-utils`, `df-derive`,
  `df-derive-core`, then the legacy `crate::core::dataframe` fallback.

## [0.3.0] - 2026-05-11

### Added

- Generic structs are now supported by `#[derive(ToDataFrame)]`, including
  default type parameters and multiple generic parameters. The macro injects
  `ToDataFrame + Columnar` bounds on each type parameter; it does not require
  user payload types to implement `Clone`.
- The unit type `()` can be used as a generic payload to contribute zero
  columns to the schema and DataFrame; direct `field: ()` fields remain
  rejected.
- Tuple-typed fields are supported, including `Option<(A, B)>`,
  `Vec<(A, B)>`, smart-pointer wrappers, and nested tuples when the outer
  tuple is not itself wrapped.
- New `#[df_derive(as_str)]` field attribute borrows string-like values via
  `AsRef<str>`, avoiding per-row `String` allocation for supported shapes.
- New `#[df_derive(as_binary)]` field attribute encodes byte-buffer shapes
  (`Vec<u8>`, `&[u8]`, and `Cow<'_, [u8]>`) as Polars `Binary` instead of the
  default `List(UInt8)`.
- New `#[df_derive(decimal(precision = N, scale = N))]` field attribute
  overrides Decimal dtype precision/scale for supported decimal backend
  shapes.
- New `#[df_derive(time_unit = "ms"|"us"|"ns")]` field attribute overrides the
  time unit for `chrono::DateTime<Tz>`, `chrono::NaiveDateTime`,
  `std::time::Duration`, `core::time::Duration`, and `chrono::Duration`.
- Chrono support now includes `chrono::DateTime<Tz>` for non-UTC time zones,
  `chrono::NaiveDateTime`, `chrono::NaiveDate`, and `chrono::NaiveTime`.
  `DateTime<Tz>` values encode the UTC instant; timezone labels are not
  preserved in the Polars dtype.
- `std::time::Duration`, `core::time::Duration`, and `chrono::Duration`
  fields are supported.
- `i128`, `u128`, and `std::num::NonZero*` integer fields are supported.
  NonZero integers encode as their underlying integer dtype.
- Borrowed reference fields are supported: `&T` peels transparently, `&str`
  is treated as a borrowed string leaf, and `&[u8]` is supported with
  `#[df_derive(as_binary)]`.
- `Box<T>`, `Rc<T>`, `Arc<T>`, and sized `Cow<'_, T>` wrappers peel
  transparently before schema and encoder selection. `Cow<'_, str>` is
  treated as a borrowed string leaf, and `Cow<'_, [u8]>` is supported with
  `#[df_derive(as_binary)]`.
- `df-derive-core` provides the canonical trait surface for non-paft
  projects, including the reference `Decimal128Encode for rust_decimal::Decimal`
  implementation.

### Changed

- **Breaking**: generated code now targets `polars` v0.53. Downstream crates
  using generated impls must use `polars = "0.53"`.
- **Breaking**: `polars-arrow` is now a required direct dependency for crates
  using `#[derive(ToDataFrame)]`. Generated code builds list arrays directly
  with Polars Arrow internals that are not re-exported from `polars::prelude`.
- **Breaking**: `ToDataFrame::schema()` now returns
  `Vec<(String, DataType)>` instead of `Vec<(&'static str, DataType)>`,
  avoiding leaked strings for nested column names.
- The `Columnar` trait now has both
  `columnar_to_dataframe(items: &[Self])` and
  `columnar_from_refs(items: &[&Self])` entry points.
- Default trait path discovery now follows `paft::dataframe`,
  `paft_utils::dataframe`, `df_derive::dataframe`,
  `df_derive_core::dataframe`, then the local `crate::core::dataframe`
  fallback.
- Decimal codegen now dispatches through `Decimal128Encode` instead of
  inlining `rust_decimal::Decimal::scale()` / `mantissa()`, so custom decimal
  backends can plug in without forking the macro. Implementations must use
  round-half-to-even on scale-down to match Polars' decimal parser.
- Generic field codegen emits trait-only paths for generic parameters, while
  concrete nested structs continue to use inherent helper fast paths.

### Fixed

- Codegen now propagates generic arguments declared on nested struct field
  types into emitted call paths, fixing nested generic fields such as
  `Outer<M> { inner: Vec<Inner<M>> }`.
- Generated code consistently routes Polars paths through the central
  `polars_paths` helper, so renamed downstream dependencies keep working.
- Bulk generic and nested-struct emitters avoid adding a `Clone` bound by using
  borrowed reference batches.

### Performance

- Bulk conversion reuses `columnar_from_refs` for generic and nested struct
  paths, avoiding per-row `DataFrame` construction and extra clones.
- String columns borrow from input rows in the columnar path, avoiding per-row
  `String` clones before Polars copies the bytes into its own buffers.
- List-output paths use typed Polars list builders instead of round-tripping
  through `AnyValue::List`, keeping inner buffers typed end to end.

## [0.2.0] - 2025-11-8

Re-release v0.1.2 under proper SemVer.

## ~~[0.1.2] - 2025-11-3~~

Yanked due to polars breaking change, use 0.2.0 instead.

### Changed

- Updated crate to support `polars` v0.52.

## [0.1.1] - 2025-09-25

### Changed

- Version bumped to 0.1.1.
- Updated crate to support `polars` v0.51.
- Internal crate resolution was updated for downstream compatibility.

## [0.1.0] - 2025-09-15

- Initial public release.

[0.3.0]: https://github.com/gramistella/df-derive/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gramistella/df-derive/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/gramistella/df-derive/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gramistella/df-derive/releases/tag/v0.1.0
