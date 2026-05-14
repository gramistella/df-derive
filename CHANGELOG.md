# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- Generated list offsets now use checked `usize` to `i64` conversions and
  return a Polars error on overflow instead of silently wrapping.
- Scalar-only numeric/bool derives with explicit custom runtime paths no
  longer require a direct `polars-arrow` dependency.
- `decimal(...)` field attributes now reject duplicate inner `precision` and
  `scale` keys instead of silently using the last value.
- Unsupported collection diagnostics now cover `BTreeSet`, `VecDeque`, and
  `LinkedList` with tailored migration hints.

## [0.3.0] - 2026-05-11

### Added

- The crate family is now split into `df-derive` (facade),
  `df-derive-core` (shared runtime trait identity), and
  `df-derive-macros` (proc macro implementation). Normal users can depend on
  `df-derive` and derive without defining local runtime traits or adding a
  runtime-path override.
- `df-derive-core` provides the canonical default runtime trait surface for
  non-paft projects, including `ToDataFrame`, `Columnar`, `ToDataFrameVec`,
  `Decimal128Encode`, the `()` payload impls, and the reference
  `Decimal128Encode for rust_decimal::Decimal` implementation.
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
- Unsupported map/set fields such as `HashMap`, `BTreeMap`, and `HashSet`
  now produce targeted diagnostics with migration hints.

### Changed

- **Breaking**: the repository root is now a workspace-only manifest, and the
  `df-derive` facade crate lives in `df-derive/` alongside
  `df-derive-core/` and `df-derive-macros/`. Path dependencies that targeted
  the repository root must target `df-derive/` instead.
- **Breaking**: generated code now targets `polars` v0.53. Downstream crates
  using generated impls must use `polars = "0.53"`.
- **Breaking**: the minimum supported Rust version is now 1.89.
- Default `df-derive` / `df-derive-core` generated code now routes Polars
  implementation dependency paths through hidden runtime re-exports, so
  downstream crates no longer need a direct `polars-arrow` dependency unless
  they use explicit custom trait-path overrides.
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
- Decimal codegen now dispatches through `Decimal128Encode` instead of
  inlining `rust_decimal::Decimal::scale()` / `mantissa()`, so custom decimal
  backends can plug in without forking the macro. Implementations must use
  round-half-to-even on scale-down to match Polars' decimal parser.
- Generic and concrete nested field codegen now uses explicit trait paths,
  improving support for qualified paths, associated types, and custom runtime
  overrides.
- `df-derive = { default-features = false }` now also disables
  `df-derive-core`'s default `rust_decimal` feature instead of enabling it
  through the facade's core dependency.
- The default runtime enables the Polars dtype feature flags required by the
  supported type matrix, including small integers, 128-bit integers, date/time,
  duration, and decimal dtypes.

### Fixed

- Codegen now propagates generic arguments declared on nested struct field
  types into emitted call paths, fixing nested generic fields such as
  `Outer<M> { inner: Vec<Inner<M>> }`.
- Generated code consistently routes Polars and Polars Arrow paths through the
  central `external_paths` helper, so renamed downstream dependencies keep
  working.
- Bulk generic and nested-struct emitters avoid adding a `Clone` bound by using
  borrowed reference batches.
- Generated `ToDataFrame` and `Columnar` impls are now marked
  `#[automatically_derived]` for better lint and tooling behavior.
- `#[df_derive(as_string)]` now adds the required `Display` bounds for custom
  struct and generic field types and propagates formatting failures as Polars
  errors.
- Generated code now uses fully-qualified standard library paths, including
  `TryFrom`, so downstream preludes and user-defined names are less likely to
  shadow generated code.
- Raw identifiers such as `r#type` are emitted as column name `type` instead
  of `r#type`.
- Nested list assembly now validates height and dtype invariants before using
  unchecked Polars constructors, turning bad manual runtime impls into Polars
  errors instead of unsound DataFrames.

### Performance

- Bulk conversion reuses `columnar_from_refs` for generic and nested struct
  paths, avoiding per-row `DataFrame` construction and extra clones.
- String columns borrow from input rows in the columnar path, avoiding per-row
  `String` clones before Polars copies the bytes into its own buffers.
- List-output paths use typed Polars builders or direct Arrow array assembly
  instead of round-tripping through `AnyValue::List`, keeping inner buffers
  typed end to end.

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
