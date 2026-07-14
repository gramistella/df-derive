# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] - Unreleased

### Added

- Wrapped nested tuple fields now compose through `Option`, `Vec`, transparent
  pointer, and nested-tuple layers instead of being rejected during terminal
  column projection.

### Changed

- **Breaking**: `Columnar::columnar_to_dataframe(&[Self])` and
  `Columnar::columnar_from_refs(&[&Self])` were replaced by
  `Columnar::encode`, which accepts any `IntoIterator<Item = &Self>`.
  Borrowed slices, reference collections, filtered inputs, and other one-shot
  iterators now share the same checked encoding boundary.
- **Breaking**: `ToDataFrame::schema()` now returns a Polars `SchemaRef`
  composed directly from the derived type instead of
  `Vec<(String, DataType)>`.
- **Breaking for manual and custom runtimes**: the derive now implements only
  hidden `ColumnarSpec::{build_schema, encode_columns}`. The selected runtime
  must provide compatible sibling `ColumnarSpec` and `ColumnSink` items, a
  checked batch boundary exposing its validated columns, and blanket
  `Columnar` and `ToDataFrame` implementations.
- Generic nested payload bounds now require only `Columnar`. A standalone
  `columnar = "..."` runtime override is accepted and its sibling
  `ColumnarSpec`, `ColumnSink`, and `Decimal128Encode` paths are inferred.
- Tuple fields retain their hierarchy through execution planning rather than
  being flattened into projection-specific terminal-column variants.

### Fixed

- The schema-bound `ColumnSink` validates column count, order, names, dtypes,
  and actual height for both generated and manual nested encoders before a
  batch can cross the public `DataFrame` boundary.
- Generated frame height now comes from the rows actually yielded by the
  caller. Iterator `size_hint()` values are allocation hints only; inaccurate
  exact-looking hints no longer misalign optional string, binary, or boolean
  validity buffers.
- Fallible primitive conversions and user-defined display or decimal work
  remain in source-row order, so an error from those operations does not
  evaluate or consume later input rows.
- Generated method parameters, schema locals, replay bindings, and deep
  tuple/list helpers are freshened against user type, const, and lifetime
  generics.
- A standalone `columnar = "..."` override now selects that runtime's sibling
  `Decimal128Encode` path instead of the default runtime's decimal trait.

### Performance

- Derived types now contain one hidden `ColumnarSpec` implementation with one
  explicit schema builder and one column encoder. Runtime blanket
  implementations provide the public single-value, slice, and iterator APIs.
- The caller's source iterator is traversed once. Shapes that need replay may
  share one buffered set of row references and use per-terminal leaf-segment
  reference buffers; safe terminal work may revisit those references after
  the source iterator is exhausted. Values are not cloned, but temporary
  pointer storage scales with yielded rows or non-empty leaf segments.
- Primitive lists use effect-aware schedules. Safe leaves choose exact-count
  deferred fills where profitable, bare deep boolean segments flatten
  contiguously and pack once, and fallible or user-defined leaves fill during
  the source pass.
- Tuple siblings share source resolution, list traversal, offsets, and
  validity. Wide safe terminals replay in bounded lanes, while fallible
  terminals preserve source evaluation order.
- `[T]::to_dataframe()` slice conversion uses its known row count directly.
  Concrete derived types cache their `SchemaRef` in a `OnceLock`; generic
  schemas remain monomorphization-dependent and are rebuilt.
- Empty structs and unit payloads preserve height without a temporary null
  column. Nested composition consumes validated child columns without
  constructing and dismantling an intermediate child `DataFrame`.
- Against the released v0.4.0 tag in isolated Criterion runs, seven of nine
  selected list benchmarks improved by approximately 3.2–24.9%; the other two
  had overlapping confidence intervals. All-safe wide tuples improved by
  49.6% and mixed Decimal tuples by 38.8%. No elapsed-time regression was
  measured.
- Nested `Vec<Vec<bool>>` improved by 14.9% in elapsed time while its Gungraun
  instruction count remained 13.4% above v0.4.0; instruction-count parity is
  therefore not claimed for that case.

### Migration

- Replace `T::columnar_to_dataframe(&rows)` with
  `rows.as_slice().to_dataframe()` for a `Vec<T>` or
  `rows.to_dataframe()` for an existing `&[T]`. These slice extensions use
  the input's known row count directly; use `T::encode(...)` for general
  iterators.
- Replace `T::columnar_from_refs(&refs)` with
  `T::encode(refs.iter().copied())`.
- Treat `T::schema()?` as a `SchemaRef` and use schema lookup or iteration
  methods instead of destructuring a `Vec<(String, DataType)>`.
- Default `value.to_dataframe()` and `slice.to_dataframe()` call sites are
  unchanged. Manual and custom runtimes must move schema and column production
  into `ColumnarSpec` and route output through their checked sink/batch
  boundary.

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
