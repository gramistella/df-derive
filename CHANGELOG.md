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
- New `#[df_derive(as_str)]` field attribute: borrows `&str` via
  `<T as AsRef<str>>::as_ref` for `T` and `Option<T>` shapes (and the
  inner-element conversion for `Vec<T>` / `Option<Vec<T>>` shapes),
  populating the same `Vec<&str>` / `Vec<Option<&str>>` columnar buffer
  the bare-`String` borrowing path uses. Use this instead of
  `#[df_derive(as_string)]` when the field's type implements
  `AsRef<str>` — it skips the per-row `String` allocation. The
  attributes are mutually exclusive on a given field; using both raises
  a compile error pointing at the field declaration. The macro emits a
  per-field `const _DF_DERIVE_ASSERT_AS_REF_STR_*` assert inside the
  generated impl so a missing `AsRef<str>` impl surfaces at the user's
  struct rather than in macro-expanded source. Performance notes: `T`,
  `Option<T>`, `Vec<T>`, and `Option<Vec<T>>` use the bulk borrowing
  path (`&[&str]` / `&[Option<&str>]` handed to Polars in one shot).
  `Vec<Vec<T>>` and `Option<Vec<Vec<T>>>` clone the inner `Vec<T>`
  per outer element (because the recursive emitter binds the inner
  Vec to a named local), then build the inner `Vec<&str>` in bulk
  for that row. `Vec<Option<T>>` and other tail-Option shapes flow
  each element through a 1-row Series plus an `AnyValue` conversion;
  no per-row `String` clone, but per-element Polars-machinery
  overhead. Stacked `Option<Option<T>>` shapes fall back to an
  allocating path that calls `AsRef<str>::as_ref` followed by
  `.to_string()` because Polars has no native
  `Option<Option<&str>>` buffer shape.

### Changed

- Updated crate to support `polars` v0.53. Generated code now calls
  `DataFrame::new_infer_height` instead of the removed `DataFrame::new(columns)`
  overload, so downstream crates must bump `polars` to `0.53`.
- Codegen now uses `Self` (instead of the bare struct ident) inside emitted
  `impl` bodies so that generated code remains valid for generic structs.
- For fields whose base type is a generic parameter, codegen emits trait-only
  paths (no crate-private inherent `__df_derive_*` calls) so that any concrete
  instantiation that satisfies `ToDataFrame + Columnar` works. Concrete nested
  structs still use the original inherent fast paths.

### Fixed

- Codegen now propagates generic arguments declared on a nested struct field
  type into the emitted call paths. Previously, a struct like
  `struct Outer<M> { inner: Vec<Inner<M>> }` would emit `Inner::schema()`,
  which the compiler couldn't resolve (`E0283: cannot satisfy '_: ToDataFrame'`)
  because `M` was unbound at the call site. The macro now emits
  `Inner::<M>::schema()` (turbofish form) and the equivalent forms for
  `columnar_to_dataframe`, `__df_derive_vec_to_inner_list_values`, and other
  inherent helpers, so any generic-parameter propagation through nested
  fields type-checks correctly.

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
- The columnar populator buffer for `String` and `Option<String>` leaf
  fields is now `Vec<&str>` / `Vec<Option<&str>>` borrowing from `items`
  instead of `Vec<String>` / `Vec<Option<String>>` cloned per row.
  `Series::new` dispatches to `StringChunked::from_slice` /
  `from_slice_options` for both shapes, producing the same
  `Utf8ViewArray`-backed column. New bench `10_string_columns` at 100k
  rows shows ~4.3× speedup for both shapes (16.1 ms → 3.70 ms required;
  15.1 ms → 3.49 ms optional). The wider `05_wide_top_level_options`
  bench (2 of 11 columns are `Option<String>`) is ~1.95× faster
  (4.09 ms → 2.13 ms). Vec-wrapped strings (`Vec<String>`,
  `Option<Vec<String>>`) and any string field with a transform
  (`as_string`, `Decimal`) keep the existing path.
- The `AnyValue`-per-row path (`__df_derive_to_anyvalues`, used by
  `Option<NestedStruct>` parents) and the row-wise `to_dataframe(&self)`
  path now also build their 1-element `Series` from `&[&str]` for
  `String` leaves with no transform, skipping the user-side clone before
  Polars copies bytes into the `Utf8ViewArray`. `03_nested_option`
  (`Vec<Owner>` where `Owner` contains `Vec<User>` and each `User` holds
  `Option<Address>` with three `String` fields) drops from 50.78 ms to
  43.72 ms (-13.9%) on top of the columnar fast path.

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
