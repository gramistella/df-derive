# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] - 2026-07-13

### Changed

- **Breaking**: `Columnar::encode` now accepts any
  `IntoIterator<Item = &Self>` and consumes that input exactly once. The
  repeatable `RowBatch` abstraction was removed, along with its separate
  reported length and repeated-traversal contract.
- **Breaking for custom runtimes**: the derive now implements hidden
  `ColumnarSpec::{build_schema, encode_columns}`. Runtimes must provide
  canonical sibling `ColumnarSpec` and `ColumnSink` items beside the selected
  `Columnar` path, a private-field `EncodedBatch`, and checked blanket
  `Columnar`/`ToDataFrame` implementations.
- `ToDataFrame::schema()` now composes and returns a Polars `SchemaRef`
  directly through `ColumnarSpec::build_schema`; it no longer discovers the
  schema by encoding an empty batch.
- Generic nested payload bounds now require only `Columnar`. A standalone
  `columnar = "..."` runtime override is accepted and its sibling
  `ColumnarSpec`, `ColumnSink`, and decimal runtime paths are inferred.

### Fixed

- Every generated or manual `ColumnarSpec` writes through a schema-bound
  `ColumnSink`. Width, ordered names, dtypes, and actual yielded height are
  checked before output becomes an `EncodedBatch`, including at the top level.
- Generated frames derive their height from the rows actually yielded by the
  caller. There is no separately reported batch length that can disagree with
  the input iterator or emitted columns.
- Generated `encode_columns` parameters and explicit-schema locals are now
  freshened against user type, const, and lifetime generics. Internal-looking
  generic names no longer collide with generated method parameters or nested
  schema composition bindings.

### Performance

- Direct, borrowed, and arbitrary iterator inputs share one generated body.
  The input is consumed once into a `Vec<&T>` so the shape-dependent column
  emitters can make their required passes without requiring a repeatable
  caller-owned batch abstraction.
- Every list column now builds its values, validity, and offsets in one scan
  of the collected rows. The separate leaf-counting pass and its generated
  counters were removed.
- Empty structs and unit payloads preserve height through the checked iterator
  boundary without a temporary null column or `drop_in_place` workaround.
- Nested composition consumes validated child batch columns directly. It no
  longer creates a child `DataFrame` merely to validate and dismantle it.
- Derived types now contain one hidden `ColumnarSpec` impl: explicit schema
  composition plus one column encoder. Public frame APIs remain blanket
  runtime methods rather than parallel generated encoding paths.

## [0.4.0] - 2026-06-29

### Changed

- **Breaking**: generated code and the default runtime now target `polars`
  v0.54 and `polars-arrow` v0.54. Downstream crates using generated impls
  should use `polars = "0.54"`.
- **Breaking**: the minimum supported Rust version is now 1.91.
- Updated tests and documentation for Polars 0.54 chunked-array iteration API
  changes.
- Refreshed the lockfile to the latest dependency graph currently resolvable
  under the workspace constraints, including `gungraun` 0.19.3 and
  `bincode-next` 3.1.1.

## [0.3.1] - 2026-05-26

### Added

- New `#[df_derive(flatten)]` field attribute splices a bare nested row
  field's schema and columns directly into the parent DataFrame without the
  field-name prefix.
- `#[df_derive(flatten(prefix = "..."))]` provides an explicit namespacing
  escape hatch while still using the flatten encoder path.
- Flattened derives now validate duplicate output column names when building
  schema and DataFrames, including duplicates from manual nested
  `ToDataFrame::schema()` implementations.

### Notes

- Flattening is intentionally narrow in this release: after transparent
  pointer peeling, no semantic wrappers may remain. Bare nested structs,
  generic row payloads, `Box<T>`, `Rc<T>`, `Arc<T>`, `&T`, and sized
  `Cow<'_, T>` are supported; nullable and list-shaped nested fields stay on
  the existing prefixed path.

## [0.3.0] - 2026-05-17

### Added

- `df-derive` is now a normal facade crate with a built-in runtime. Most
  projects can depend on `df-derive`, import `df_derive::prelude::*`, and use
  `#[derive(ToDataFrame)]` without defining local runtime traits or adding
  `#[df_derive(trait = "...")]`.
- New `df-derive-core` and `df-derive-macros` crates are available for users
  who want the shared runtime traits separately from the proc macro.
  `df-derive-core` provides `ToDataFrame`, `Columnar`, `ToDataFrameVec`,
  `Decimal128Encode`, the `()` payload impls, and the default
  `rust_decimal::Decimal` decimal encoder.
- Generic structs are now supported by `#[derive(ToDataFrame)]`, including
  default type parameters and multiple generic parameters. The macro injects
  bounds by role (`ToDataFrame + Columnar`, `AsRef<str>`, `Display`, or
  `Decimal128Encode`) and does not require generic payload types to implement
  `Clone`.
- The unit type `()` can be used as a generic payload to contribute zero
  columns to the schema and DataFrame; direct `field: ()` fields remain
  rejected.
- Tuple-typed fields are supported, including `Option<(A, B)>`,
  `Vec<(A, B)>`, smart-pointer wrappers, and unwrapped nested tuples. Wrapped
  nested tuple projection paths are rejected with an error.
- New `#[df_derive(skip)]` field attribute omits a field from generated schema
  and DataFrame output, including unsupported helper fields and tuple struct
  fields.
- New `#[df_derive(as_str)]` field attribute borrows string-like values via
  `AsRef<str>`, avoiding per-row `String` allocation for supported shapes.
- New `#[df_derive(as_binary)]` field attribute encodes byte-buffer shapes
  (`Vec<u8>`, `&[u8]`, and `Cow<'_, [u8]>`) as Polars `Binary` instead of the
  default `List(UInt8)`.
- New `#[df_derive(decimal(precision = N, scale = N))]` field attribute
  overrides Decimal dtype precision/scale and lets custom decimal backends opt
  into Polars decimal columns through `Decimal128Encode`.
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
- More unsupported shapes now produce targeted diagnostics with migration
  hints, including maps, sets, `VecDeque`, `LinkedList`, mutable references,
  unsized smart-pointer leaves, recursive nested fields, and ambiguous bare
  `Duration` fields.

### Changed

- **Breaking**: the repository root is now a workspace-only manifest, and the
  `df-derive` facade crate lives in `df-derive/` alongside
  `df-derive-core/` and `df-derive-macros/`. Path dependencies that targeted
  the repository root must target `df-derive/` instead.
- **Breaking**: generated code now targets `polars` v0.53. Downstream crates
  using generated impls must use `polars = "0.53"`.
- **Breaking**: the minimum supported Rust version is now 1.90.
- Default `df-derive` / `df-derive-core` generated code now routes Polars
  implementation dependency paths through hidden runtime re-exports, so
  downstream crates no longer need a direct `polars-arrow` dependency unless
  they use explicit custom trait-path overrides.
- Explicit `df_derive::dataframe::ToDataFrame` and
  `df_derive_core::dataframe::ToDataFrame` trait-path overrides are treated as
  the default runtime and keep using hidden runtime re-exports.
- Custom runtimes selected with explicit `#[df_derive(trait = "...")]`
  overrides still need compatible direct `polars` and `polars-arrow`
  dependencies, because generated code builds typed list arrays directly.
- **Breaking**: `ToDataFrame::schema()` now returns
  `Vec<(String, DataType)>` instead of `Vec<(&'static str, DataType)>`,
  avoiding leaked strings for nested column names.
- **Breaking for custom runtimes**: the `Columnar` trait now has both
  `columnar_to_dataframe(items: &[Self])` and
  `columnar_from_refs(items: &[&Self])` entry points.
- Default runtime discovery now checks `df_derive::dataframe`,
  `df_derive_core::dataframe`, `paft_utils::dataframe`,
  `paft::dataframe`, then the local `crate::core::dataframe`
  fallback.
- Container-level `#[df_derive(...)]` runtime overrides now reject duplicate
  keys, and `columnar = "..."` is rejected unless it is paired with
  `trait = "..."` to avoid mixed-runtime impls.
- Decimal fields now encode through `Decimal128Encode` instead of being tied
  to `rust_decimal::Decimal::scale()` / `mantissa()`, so custom decimal
  backends can plug in without forking the macro. Implementations must use
  round-half-to-even on scale-down to match Polars' decimal parser.
- Generic and concrete nested fields now use explicit trait paths, improving
  support for qualified paths, associated types, and custom runtime overrides.
- `df-derive = { default-features = false }` now also disables
  `df-derive-core`'s default `rust_decimal` feature instead of enabling it
  through the facade's core dependency.
- The default runtime enables the Polars dtype feature flags required by the
  supported type matrix, including small integers, 128-bit integers, date/time,
  duration, and decimal dtypes.
- Scalar-only numeric and bool derives using custom runtime paths no longer
  require a direct `polars-arrow` dependency.

### Fixed

- Nested struct fields now preserve their declared generic arguments in
  generated call paths, fixing nested generic fields such as
  `Outer<M> { inner: Vec<Inner<M>> }`.
- Qualified type paths, associated type paths, renamed dependencies, and
  explicit built-in runtime paths are handled more reliably in generated code.
- Bulk generic and nested-struct conversions avoid adding unnecessary `Clone`
  bounds.
- Generated `ToDataFrame` and `Columnar` impls are now marked
  `#[automatically_derived]` for better lint and tooling behavior.
- `#[df_derive(as_string)]` now adds the required `Display` bounds for custom
  struct and generic field types, rejects non-displayable concrete fields
  earlier, and propagates formatting failures as Polars errors.
- `#[df_derive(as_str)]` validates `AsRef<str>` requirements for concrete,
  generic, and smart-pointer fields before compilation proceeds.
- Generated code now uses fully-qualified standard library paths, including
  `TryFrom`, so downstream preludes and user-defined names are less likely to
  shadow generated code.
- Raw identifiers such as `r#type` are emitted as column name `type` instead
  of `r#type`.
- Nested list construction now validates offsets, lengths, heights, dtypes,
  and decimal precision before constructing Polars arrays. Overflow,
  truncation, bad manual runtime impls, and invalid decimal scales now return
  Polars errors instead of panicking, wrapping, or producing invalid output.
- Container and field attributes now reject duplicate keys and incompatible
  combinations instead of silently accepting the last value or producing mixed
  runtime impls.
- Enum and union derive targets, wrapped nested tuples, direct `()` fields,
  invalid binary fields, invalid time-unit fields, and invalid decimal fields
  now fail with clearer diagnostics.
- `df-derive = { default-features = false }` no longer pulls in the default
  `rust_decimal` support through `df-derive-core`.

### Performance

- Batch conversion for nested structs, generic fields, and list-heavy shapes
  avoids per-row `DataFrame` construction and unnecessary clones.
- String and `#[df_derive(as_str)]` columns borrow from input rows during
  column construction instead of cloning each value first.
- Decimal columns encode through `i128` mantissas instead of formatting values
  through strings.
- List-output paths use typed Polars builders or direct Arrow array assembly
  instead of round-tripping through `AnyValue::List`, improving nested
  `Vec<T>`, `Vec<Option<T>>`, `Vec<Vec<T>>`, and `Vec<Struct>` conversions.

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

[0.5.0]: https://github.com/gramistella/df-derive/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/gramistella/df-derive/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/gramistella/df-derive/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/gramistella/df-derive/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gramistella/df-derive/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/gramistella/df-derive/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gramistella/df-derive/releases/tag/v0.1.0
