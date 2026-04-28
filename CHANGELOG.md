# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-04-28

### Added

- Generic structs are now supported by `#[derive(ToDataFrame)]`. The macro
  injects `ToDataFrame + Columnar + Clone` bounds on every type parameter
  using `split_for_impl`, so any concrete instantiation must satisfy those
  traits. `Clone` is required because the bulk emitters collect a
  contiguous `Vec<T>` from `&[Self]` before delegating to
  `T::columnar_to_dataframe`; injecting it at the macro level surfaces a
  missing bound at the user's struct definition rather than deep inside
  macro-expanded source.
- Default type parameters and multiple generic parameters are supported.
- The unit type `()` can now be used as a generic payload to contribute zero
  columns to the schema and DataFrame; reference impls of `ToDataFrame` and
  `Columnar` for `()` are provided in `tests/common.rs`.
- New benchmark `09_generics` covering unit, primitive, and nested-struct
  generic instantiations, plus an A/B comparison between the per-row and
  bulk columnar paths for generic fields.

### Changed

- Codegen now uses `Self` (instead of the bare struct ident) inside emitted
  `impl` bodies so that generated code remains valid for generic structs.
- For fields whose base type is a generic parameter, codegen emits trait-only
  paths (no crate-private inherent `__df_derive_*` calls) so that any concrete
  instantiation that satisfies `ToDataFrame + Columnar` works. Concrete nested
  structs still use the original inherent fast paths.

### Performance

- The columnar path now flattens generic-leaf fields by collecting a
  `Vec<T>` once and calling `<T as Columnar>::columnar_to_dataframe` exactly
  once, then prefix-renaming the resulting columns. Compared to the per-row
  fallback that built one tiny `DataFrame` per item, this is roughly 17× to
  140× faster at 100k rows depending on `T` (see `benches/09_generics`).
- The same bulk strategy now applies to the helpers' vec-anyvalues path
  (`__df_derive_vec_to_inner_list_values`), which is invoked when an outer
  struct contains `Vec<Wrapper<T>>`. At 100k rows this gives ~120× speedup
  for primitive `T` and ~20× for nested-struct `T`.
- The bulk path now also covers `Option<T>` and `Vec<T>` directly (depth-1
  wrappers around a generic parameter): `Option<T>` calls
  `T::columnar_to_dataframe` once over the gathered `Some` values and
  scatters columns back with `AnyValue::Null` at `None` positions; `Vec<T>`
  flattens all parent rows' inner vectors into a single contiguous slice
  with offsets, calls `T::columnar_to_dataframe` once, and slices each
  column per parent row. Bench at 100k rows: ~16× faster for
  `OptWrap<Meta>`, ~9× faster for `VecWrap<Meta>`. Deeper compositions like
  `Option<Option<T>>` or `Vec<Vec<T>>` keep the per-row trait-only fallback.

## [0.2.0] - 2025-11-8

Re-release v0.1.2 under proper SemVer

## ~~[0.1.2] - 2025-11-3~~

Yanked due to polars breaking change, use 0.2.0 instead

### Changed

- Updated crate to support `polars` v0.52

## [0.1.1] - 2025-09-25

### Changed

- Version bumped to 0.1.1
- Updated crate to support `polars` v0.51
- Internal: Updated crate resolution from `paft-core` to `paft-utils` for internal downstream crate compatibility

## [0.1.0] - 2025-09-15

- Initial public release.

[0.3.0]: https://github.com/gramistella/df-derive/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gramistella/df-derive/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/gramistella/df-derive/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gramistella/df-derive/releases/tag/v0.1.0
