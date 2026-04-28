# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-04-28

### Added

- Generic structs are now supported by `#[derive(ToDataFrame)]`. The macro
  injects `ToDataFrame + Columnar` bounds on every type parameter using
  `split_for_impl`, so any concrete instantiation must satisfy those traits.
- Default type parameters and multiple generic parameters are supported.
- The unit type `()` can now be used as a generic payload to contribute zero
  columns to the schema and DataFrame; reference impls of `ToDataFrame` and
  `Columnar` for `()` are provided in `tests/common.rs`.
- New benchmark `09_generics` covering unit, primitive, and nested-struct
  generic instantiations.

### Changed

- Codegen now uses `Self` (instead of the bare struct ident) inside emitted
  `impl` bodies so that generated code remains valid for generic structs.
- For fields whose base type is a generic parameter, codegen emits trait-only
  paths (no crate-private inherent `__df_derive_*` calls) so that any concrete
  instantiation that satisfies `ToDataFrame + Columnar` works. Concrete nested
  structs still use the original inherent fast paths.

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
