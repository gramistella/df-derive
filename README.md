# df-derive

[![Crates.io](https://img.shields.io/crates/v/df-derive.svg)](https://crates.io/crates/df-derive)
[![Docs.rs](https://docs.rs/df-derive/badge.svg)](https://docs.rs/df-derive)
[![CI](https://github.com/gramistella/df-derive/actions/workflows/ci.yml/badge.svg)](https://github.com/gramistella/df-derive/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/crates/d/df-derive)](https://crates.io/crates/df-derive)
[![License](https://img.shields.io/crates/l/df-derive)](LICENSE)

`df-derive` derives fast conversions from Rust structs into Polars
`DataFrame`s. The normal user-facing crate now includes a default runtime
trait surface, so most projects can write `#[derive(ToDataFrame)]` without a
local trait module or `#[df_derive(trait = "...")]` override.

## What This Crate Does

Deriving `ToDataFrame` on structs and tuple structs generates
allocation-conscious code to:

- Convert a single value to a `polars::prelude::DataFrame`
- Convert slices through a columnar batch path
- Inspect generated column names and `DataType`s through `T::schema()`

The derive supports nested structs flattened with dot notation, nullable
shapes with `Option<T>`, list shapes with `Vec<T>`, tuple structs,
tuple-typed fields, generic structs, borrowed fields, smart pointers, datetime
types, duration types, byte blobs, and decimal backends.

## Quick Start

```toml
[dependencies]
df-derive = "0.4"
polars = "0.54"

# If your models use these types:
chrono = { version = "0.4", features = ["serde"] }
rust_decimal = { version = "1.42", default-features = false, features = ["std"] }
```

With the default `df-derive` facade, generated impls use hidden runtime
re-exports for implementation details such as `polars-arrow`; downstream
crates do not need to depend on `polars-arrow` directly. Keep `polars` direct
when your code names Polars types. The default runtime enables the Polars
dtype features required by the supported matrix below.

```rust
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Trade {
    symbol: String,
    price: f64,
    size: u64,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![
        Trade { symbol: "AAPL".into(), price: 187.23, size: 100 },
        Trade { symbol: "MSFT".into(), price: 411.61, size: 200 },
    ];

    let df = rows.as_slice().to_dataframe()?;
    println!("{df}");
    Ok(())
}
```

The default runtime API is available as `df_derive::dataframe::*`. The prelude
exports the derive macro plus `ToDataFrame`, `Columnar`, `ToDataFrameVec`, and
`Decimal128Encode`; it also exports the trait as
`ToDataFrameTrait` for code that wants an unambiguous type-namespace alias.

## Benchmarks

A reproducible head-to-head benchmark comparing `df-derive` with hand-written
Polars, `serde_arrow`, and `polars-row-derive` lives in
[comparison-bench/README.md](comparison-bench/README.md). Its historical
df-derive 0.3.1 / Polars 0.53 report is
[comparison-bench/BENCHMARK.md](comparison-bench/BENCHMARK.md).

## Crate Layout

This repository uses a serde-like three-crate architecture:

- `df-derive`: the normal facade crate. It re-exports the derive macro from
  `df-derive-macros` and the runtime API from `df-derive-core`.
- `df-derive-core`: a normal library crate that owns the shared public runtime
  traits, the hidden `ColumnarSpec`/checked `ColumnSink` composition boundary,
  unit-payload support, and the optional reference
  `Decimal128Encode for rust_decimal::Decimal` impl.
- `df-derive-macros`: the proc-macro implementation. Power users can depend
  on this directly and target `df-derive-core`, `paft`, or a custom runtime.
  Encoder lowering composes a phase-typed plan with a single caller-source
  scan and, for selected shapes, ordered deferred phases before rendering
  generated Rust.

Because `df-derive-core` owns the default trait identity, models derived in
different crates can compose as nested `ToDataFrame` types when they use the
facade/default runtime.

## Generated API

For each struct or tuple struct `T`, the macro generates one hidden runtime
implementation:

```rust,ignore
impl ColumnarSpec for T {
    const REQUIRES_ROW_REPLAY: bool = /* generated policy */;

    fn build_schema() -> PolarsResult<SchemaRef>;

    fn encode_columns<'a, I>(
        rows: &mut I,
        sink: &mut ColumnSink,
    ) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>;
}
```

`build_schema` composes names and dtypes directly; schema inspection no longer
encodes an empty batch. `encode_columns` writes through schema-bound slots.
`RowCursor` is a hidden sibling runtime trait extending `Iterator` with exact
yielded-row counting and replay operations. Its ordering and counting rules
are semantic rather than memory-safety requirements because runtime-owned
storage validates generated fill boundaries. Each generated
`ColumnarSpec` advertises whether it requires row replay. The blanket general-
iterator boundary selects a compact streaming cursor for non-replaying shapes
and a reference-buffering cursor for replaying shapes; generated replaying
encoders enable capture before consuming their first row. Slice-backed and
nested reference-slice cursors replay their yielded prefix directly without
another row-reference buffer. The runtime blanket APIs own the caller's input:
`Columnar::encode` counts rows from a general one-shot iterator, while the
slice extension forwards the slice's known length directly. Each
`ColumnSink::next_slot` exposes the authoritative output name and dtype, and
`ColumnSlot::commit` checks the materialized column before advancing. The sink
then checks width and every column height before producing
an `EncodedBatch`. Public conversion constructs the outer `DataFrame`; nested
encoders consume a child batch's validated columns directly without
constructing and dismantling a child frame. `ToDataFrame` and the slice
extension remain blanket APIs. Primitive-list encoders allocate and patch
exact storage through the selected runtime's hidden
`__private::encode::{CapturedSegments, CapturedSegmentGroups, ExactBuffer,
PreparedValidity, PreparedBooleanValues}` types. The captured schedules retain
checked aggregate lengths and the storage types own exact vector and bitmap
fills, so generated implementations do not contain unchecked writes.

## Representative Generated Code

For the quick-start `Trade` struct, the derive emits code shaped like this.
The snippet is abridged for readability: generated dependency paths are
shortened with imports, rustc's `vec!` expansion is omitted, and
compiler-generated helper blocks are removed.

```rust,ignore
use df_derive::dataframe::{ColumnSink, ColumnarSpec, RowCursor};
use df_derive::dataframe::__private::{
    polars::prelude::{
        DataType, Float64Chunked, IntoSeries, PolarsResult, Schema, SchemaRef,
        StringChunked, UInt64Chunked,
    },
    polars_arrow::array::MutableBinaryViewArray,
};

#[automatically_derived]
impl ColumnarSpec for Trade {
    const REQUIRES_ROW_REPLAY: bool = false;

    fn build_schema() -> PolarsResult<SchemaRef> {
        let fields = [
            ("symbol".into(), DataType::String),
            ("price".into(), DataType::Float64),
            ("size".into(), DataType::UInt64),
        ];
        Ok(std::sync::Arc::new(
            Schema::from_iter_check_duplicates(fields)?,
        ))
    }

    fn encode_columns<'a, I>(
        rows: &mut I,
        sink: &mut ColumnSink,
    ) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let row_capacity = rows.size_hint().0;
        let mut symbol = MutableBinaryViewArray::<str>::with_capacity(row_capacity);
        let mut price = Vec::<f64>::with_capacity(row_capacity);
        let mut size = Vec::<u64>::with_capacity(row_capacity);

        for item in rows.by_ref() {
            symbol.push_value_ignore_validity(item.symbol.as_str());
            price.push(item.price);
            size.push(item.size);
        }

        let slot = sink.next_slot()?;
        let s = IntoSeries::into_series(StringChunked::with_chunk(
            slot.name().clone(),
            symbol.freeze(),
        ));
        slot.commit(s.into())?;

        let slot = sink.next_slot()?;
        let s = IntoSeries::into_series(Float64Chunked::from_vec(slot.name().clone(), price));
        slot.commit(s.into())?;

        let slot = sink.next_slot()?;
        let s = IntoSeries::into_series(UInt64Chunked::from_vec(slot.name().clone(), size));
        slot.commit(s.into())
    }
}
```

## Supported Types And Shapes

Container and wrapper support:

- **Named structs**: each field becomes one or more columns.
- **Nested structs**: fields flatten recursively with dot notation.
- **Explicit field flattening**: `#[df_derive(flatten)]` on a bare nested row
  field splices the child columns into the parent without the field-name
  prefix.
- **Vec of primitives and structs**: `Vec<T>` becomes a Polars `List` column;
  `Vec<Nested>` becomes one list column per nested field.
- **`Option<T>`**: scalar and list columns carry null validity.
- **Tuple structs**: unnamed fields become `field_0`, `field_1`, and so on.
- **Tuple-typed fields**: `pair: (A, B)` flattens to
  `pair.field_0`, `pair.field_1`; `Option<(A, B)>` and `Vec<(A, B)>`
  distribute the outer wrapper across the element columns.
- **Empty structs**: an instance produces shape `(1, 0)` and an empty slice
  produces shape `(0, 0)`.
- **Generics**: generic structs are supported; the macro injects the
  necessary `Columnar` bounds, plus `Decimal128Encode` for
  generic parameters annotated with `decimal(...)`.
- **Transparent pointers**: `Box<T>`, `Rc<T>`, `Arc<T>`, borrowed references
  `&T`, and `Cow<'_, T>` with a sized inner peel transparently and preserve
  the bare field's column shape and dtype.

Common leaf types:

- **Primitives**: `String`, `&str`, `bool`, signed and unsigned integer types
  including `i128`/`u128` and `isize`/`usize`, `std::num::NonZero*` integer
  types, `f32`, and `f64`.
- **Time**: `chrono::DateTime<Tz>` and `chrono::NaiveDateTime` encode as
  `Datetime(Milliseconds, None)` by default; use
  `#[df_derive(time_unit = "ms" | "us" | "ns")]` to override.
  `DateTime<Tz>` values are encoded as UTC instants, so use
  `#[df_derive(as_string)]` if the textual timezone or offset matters.
- **Date and time-of-day**: `chrono::NaiveDate` encodes as `Date`, and
  `chrono::NaiveTime` encodes as `Time`. These encodings are fixed and do not
  accept `time_unit`.
- **Duration**: `std::time::Duration`, `core::time::Duration`, and
  `chrono::Duration` encode as `Duration(Nanoseconds)` by default; use
  `time_unit` to choose milliseconds, microseconds, or nanoseconds. Bare
  `Duration` is rejected as ambiguous.
- **Decimal**: bare `Decimal` and `rust_decimal::Decimal` encode as
  `Decimal(38, 10)` by default. Custom decimal backends opt in with
  `#[df_derive(decimal(precision = N, scale = S))]`.
- **Binary blobs**: `#[df_derive(as_binary)]` opts `Vec<u8>`, `&[u8]`, or
  `Cow<'_, [u8]>` shapes into Polars `Binary`; unannotated `Vec<u8>` remains
  `List(UInt8)`.

## Dtype Support Matrix

The default `df-derive` facade and `df-derive-core` runtime enable the Polars
features in this table on their `polars` dependency. If you use
`df-derive-macros` with a custom runtime and no `df-derive-core` dependency,
enable the matching features on that runtime's direct `polars` dependency.

| Rust leaf family | Polars dtype emitted | Polars feature for custom runtimes |
| --- | --- | --- |
| `bool` | `Boolean` | none |
| `String`, `&str`, `as_str`, `as_string` | `String` | none |
| `i8`, `NonZeroI8` | `Int8` | `dtype-i8` |
| `i16`, `NonZeroI16` | `Int16` | `dtype-i16` |
| `i32`, `i64`, `isize`, matching `NonZero*` | `Int32` / `Int64` | none |
| `i128`, `NonZeroI128` | `Int128` | `dtype-i128` |
| `u8`, `NonZeroU8` | `UInt8` | `dtype-u8` |
| `u16`, `NonZeroU16` | `UInt16` | `dtype-u16` |
| `u32`, `u64`, `usize`, matching `NonZero*` | `UInt32` / `UInt64` | none |
| `u128`, `NonZeroU128` | `UInt128` | `dtype-u128` |
| `f32`, `f64` | `Float32` / `Float64` | none |
| `chrono::DateTime<Tz>`, `chrono::NaiveDateTime` | `Datetime` | `dtype-datetime`, plus `timezones` for timezone-aware values |
| `chrono::NaiveDate` | `Date` | `dtype-date` |
| `chrono::NaiveTime` | `Time` | `dtype-time` |
| `std::time::Duration`, `core::time::Duration`, `chrono::Duration` | `Duration` | `dtype-duration` |
| `Decimal`, `rust_decimal::Decimal`, custom decimal backends | `Decimal` | `dtype-decimal` |
| `#[df_derive(as_binary)]` byte buffers | `Binary` | none |

`Option<T>`, `Vec<T>`, tuples, and nested structs preserve the leaf dtype;
each `Vec` layer wraps the leaf in `List(...)`.

For Polars 0.54, `dtype-decimal` enables the decimal column machinery and its
internal `Int128` backing path. You only need an explicit `dtype-i128` feature
when your derived structs expose `i128` / `NonZeroI128` fields as `Int128`
columns.

Useful field attributes:

- `#[df_derive(skip)]`: omit a field from generated schema and DataFrame output.
- `#[df_derive(flatten)]`: splice a bare nested row field into the parent without the field-name prefix.
- `#[df_derive(flatten(prefix = "..."))]`: splice a bare nested row field with an explicit output namespace.
- `#[df_derive(as_string)]`: format values with `Display` into a string column using a reused scratch buffer.
- `#[df_derive(as_str)]`: borrow via `AsRef<str>` without `Display` formatting or an intermediate scratch buffer.
- `#[df_derive(as_binary)]`: encode byte-buffer shapes as Binary.
- `#[df_derive(decimal(precision = N, scale = S))]`: choose a decimal dtype or opt a custom decimal backend into `Decimal128Encode`.
- `#[df_derive(time_unit = "ms" | "us" | "ns")]`: choose datetime or duration units.

`skip` is useful for caches, source metadata, handles, or unsupported helper
fields that should remain on the Rust struct but not become DataFrame columns.
It is mutually exclusive with conversion attributes because skipped fields are
not analyzed or emitted. Tuple struct fields can be skipped too; remaining
tuple columns keep their original `field_{index}` names.

`flatten` is useful for reusable key/value row structs whose fields should
appear at the parent table level. It is accepted only for bare nested row
shapes after transparent pointer peeling, such as `Key`, `Box<Key>`,
`Arc<Key>`, `&Key`, or a bare generic row payload. `Option<Key>`,
`Vec<Key>`, and other semantic wrappers remain on the normal prefixed nested
path. Flattened derives validate duplicate output names when building schema
and DataFrames. Use `flatten(prefix = "...")` when intentional namespacing is
needed.

`as_string` is useful for enums or validated newtypes that should appear as
string columns. It formats each value into a reusable `String` scratch buffer
before pushing the resulting `&str` into the column builder; the builder still
copies bytes into the output column, and the scratch can grow to fit the
largest formatted value. If a field already implements `AsRef<str>`, prefer
`as_str`: it borrows through the same columnar buffer used for bare
`String`/`&str` fields and skips both `Display` formatting and the scratch
buffer. The two attributes are mutually exclusive.

`as_binary` accepts `Vec<u8>`, `Option<Vec<u8>>`, `Vec<Vec<u8>>`,
`Vec<Option<Vec<u8>>>`, `Option<Vec<Vec<u8>>>`, and the same shapes over
`&[u8]` and `Cow<'_, [u8]>`. Bare `u8`, `Option<u8>`,
`Vec<Option<u8>>`, non-`u8` leaves, and `String` are rejected. The binary
attribute is mutually exclusive with `as_str`, `as_string`, `decimal(...)`,
and `time_unit`.

Enums and unions are not supported as derive targets; use `as_string` or
`as_str` on enum fields. Direct fields of type `()` are rejected, but `()` is
supported as a generic payload and contributes zero columns.

Tuple fields cannot carry field-level conversion attributes such as `as_str`,
`as_binary`, `decimal(...)`, or `time_unit`; hoist that value into a named
struct when you need an attributed field. Tuple hierarchy and wrappers compose,
including nested tuples inside `Option` and `Vec` layers.

## Column Naming

- Named struct fields use the Rust field name, such as `symbol`.
- Nested structs use dot notation recursively, such as `address.city`.
- `#[df_derive(flatten)]` nested fields omit the parent field name, such as
  `city` instead of `address.city`.
- `#[df_derive(flatten(prefix = "home"))]` nested fields use the explicit
  prefix, such as `home.city`.
- `Vec<Nested>` fields use the outer field plus nested field name, such as
  `quotes.close`.
- Tuple-typed fields use `field.field_0`, `field.field_1`, and recurse for
  nested tuples regardless of their `Option` and `Vec` wrappers.
- Tuple structs use `field_0`, `field_1`, and so on.

## Limitations And Guidance

- Maps such as `HashMap<_, _>` and `BTreeMap<_, _>` are not supported; use
  `Vec<(K, V)>` or a named row struct when you need a tabular representation.
- Sets such as `HashSet<_>` and `BTreeSet<_>` are not supported; use
  `Vec<T>` when you need a list representation.
- Sequence collections such as `VecDeque<T>` and `LinkedList<T>` are not
  supported; use `Vec<T>` instead.
- All nested custom structs must implement `Columnar`, normally by also
  deriving `ToDataFrame`.
- Obvious direct self-recursive nested fields using `Self`, the bare deriving
  type name, `self::Type`, or `crate::Type` are rejected after transparent
  wrapper peeling, including shapes such as `Node`, `Box<Node>`,
  `Option<Box<Node>>`, and tuple fields containing the same. Use identifier
  fields or a separate flat representation for recursive data structures.
- Recursive nested row schemas are unsupported, including qualified or
  mutually recursive cycles hidden behind pointer wrappers. Explicit schema
  composition still recurses through the nested type graph, so a cycle has no
  terminating child. Use identifiers or a separate acyclic representation.
- Consecutive `Option` layers above a `Vec` collapse to one list-level
  validity bit, so `None` and `Some(None)` are indistinguishable in the
  resulting list column.
- Borrowed byte slices and `Cow<'_, [u8]>` require `#[df_derive(as_binary)]`;
  other borrowed slice forms are rejected. Use `Vec<T>` for list columns.

## Runtime Discovery And Overrides

Explicit container attributes always win:

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(
    trait = "my_runtime::dataframe::ToDataFrame",
    columnar = "my_runtime::dataframe::Columnar",
    decimal128_encode = "my_runtime::dataframe::Decimal128Encode",
)]
struct Row {
    amount: MyDecimal,
}
```

If only `trait = "x::ToDataFrame"` is provided, the macro infers the sibling
`x::Columnar` and `x::Decimal128Encode` paths. From the resolved `Columnar`
path it also requires canonical sibling `ColumnarSpec` and `ColumnSink` items,
plus `x::__private::encode` when primitive-list storage is generated.
The checked runtime surface conventionally exposes `EncodedBatch` there too.
A standalone
`columnar = "x::Columnar"` override is also supported; the macro infers its
sibling `ColumnarSpec`, `ColumnSink`, and `Decimal128Encode` paths. The
runtime's blanket `ToDataFrame` API keeps its own trait identity.

Explicit paths to the built-in facade/core runtimes,
`df_derive::dataframe::ToDataFrame` or
`df_derive_core::dataframe::ToDataFrame` (including dependency renames), still
use the default-runtime dependency roots from that same `dataframe` module's
hidden `__private` re-exports. They do not require a direct `polars-arrow`
dependency just because the trait path was written explicitly.

Explicit `trait` + `columnar` pairs also cannot mix the built-in
`df_derive`/`df_derive_core` dataframe runtime with a custom runtime. Use the
matching built-in `Columnar` path, omit `columnar` so it is inferred from the
built-in `trait`, or provide a fully custom pair.

Without overrides, the macro discovers a `dataframe` module in this order:

1. `df_derive::dataframe`
2. `df_derive_core::dataframe`
3. `paft_utils::dataframe`
4. `paft::dataframe`
5. `crate::core::dataframe`

Discovery uses `proc_macro_crate::crate_name`, so dependency renames are
respected. For example, a dependency declared as
`dfd = { package = "df-derive", version = "0.4" }` is emitted as
`::dfd::dataframe`.

The final `crate::core::dataframe` fallback is for legacy/local runtimes in
crates that use `df-derive-macros` directly without `df-derive`,
`df-derive-core`, `paft-utils`, or `paft`. Any runtime reached by this default
discovery path must expose `dataframe::__private::{polars, polars_arrow,
encode}` for generated-code dependency roots and runtime-owned primitive-list
storage.

## Power-User Runtime Choices

Use the facade for the default runtime:

```rust
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    id: u32,
}
```

Use the macro crate directly with the shared core runtime:

```toml
[dependencies]
df-derive-core = "0.4"
df-derive-macros = "0.4"
polars = "0.54"
```

```rust
use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
use df_derive_macros::ToDataFrame;

#[derive(ToDataFrame)]
struct Row {
    id: u32,
}
```

Use a custom runtime by providing compatible traits and overriding paths.
Outside the built-in facade/core paths described above, custom runtimes
selected with `#[df_derive(trait = "...")]` must name a compatible direct
`polars` dependency. They also need a compatible direct `polars-arrow`
dependency when the derived fields use shapes that require generated Arrow
array builders, such as list, nullable primitive, string, or binary columns.
Scalar-only numeric/bool derives do not need `polars-arrow`.

A custom trait identity must own its checked boundary: compatible
`RowCursor` and `ColumnarSpec` traits; blanket `Columnar` and `ToDataFrame`
traits; a `ColumnSink` with public `next_slot`; a returned `ColumnSlot` with
public `name`, `dtype`, and consuming `commit` operations;
and an `EncodedBatch` whose generated-code operation is public
`into_columns`. `Columnar` must provide the general iterator boundary plus the
specialized slice and reference-slice boundaries used by generated nested
encoders. Sink construction and finalization stay private to the blanket
`Columnar` implementation, so a manual `ColumnarSpec` cannot swap in a sink
bound to a different schema. The derive sets `REQUIRES_ROW_REPLAY`
automatically; a manual `ColumnarSpec` that invokes `enable_replay` or `replay`
must set it to `true` so the general-iterator boundary selects the buffering
cursor. The
[compile-checked local runtime fixture](df-derive/tests/support/local_runtime.rs)
shows the re-export strategy. Custom runtimes with a different Polars identity
must implement the same hidden `__private::encode` surface; an independent
implementation is compiled and executed by the
[architecture fixture](df-derive/tests/architecture.rs). The required methods
are deliberately narrow:

```text
ExactBuffer<T>:
  with_capacity(usize), with_exact_len(usize),
  extend_segment(&[S], map: &S -> T),
  extend_captured(&CapturedSegments<S>, map: &S -> T),
  extend_grouped(&CapturedSegmentGroups<S>, map: &S -> T) -> Vec<i64>,
  try_extend_segment(&[S], map: &S -> Result<T, E>),
  extend_nullable_segment(
    &mut PreparedValidity, &[S], map: &S -> (T, bool)
  ),
  extend_nullable_captured(
    &mut PreparedValidity, &CapturedSegments<S>, map: &S -> Option<T>
  ) where T: Default,
  extend_nullable_grouped(
    &mut PreparedValidity, &CapturedSegmentGroups<S>, map: &S -> Option<T>
  ) -> Vec<i64> where T: Default,
  try_extend_nullable_segment(
    &mut PreparedValidity, &[S], map: &S -> Result<(T, bool), E>
  ),
  len(), finish() -> Vec<T>
PreparedValidity:
  with_capacity(usize), with_exact_len(usize),
  extend_segment(values_len, &[S], map: &S -> bool),
  extend_captured(values_len, &CapturedSegments<S>, map: &S -> bool),
  extend_grouped(
    values_len, &CapturedSegmentGroups<S>, map: &S -> bool
  ) -> Vec<i64>,
  try_extend_segment(values_len, &[S], map: &S -> Result<bool, E>),
  finish(actual_len) -> Option<Bitmap>
PreparedBooleanValues:
  with_exact_len(usize),
  extend_segment(&[S], map: &S -> bool),
  extend_captured(&CapturedSegments<S>, map: &S -> bool),
  extend_grouped(&CapturedSegmentGroups<S>, map: &S -> bool) -> Vec<i64>,
  extend_nullable_segment(
    &mut PreparedValidity, &[S], map: &S -> (bool, bool)
  ),
  extend_nullable_captured(
    &mut PreparedValidity, &CapturedSegments<S>, map: &S -> Option<bool>
  ),
  extend_nullable_grouped(
    &mut PreparedValidity, &CapturedSegmentGroups<S>, map: &S -> Option<bool>
  ) -> Vec<i64>,
  len(), finish() -> Bitmap
CapturedSegments<'a, S>:
  new(), with_capacity(usize), capture(&'a Vec<S>) -> Option<usize>,
  len(), visit(map: &S -> ())
CapturedSegmentGroups<'a, S>:
  with_capacity(usize), capture_group(&'a Vec<Vec<S>>) -> Option<()>,
  len(), segment_count(),
  visit_segments(map: &Vec<S> -> ()) -> Vec<i64>
```

All constructors, schedule captures, and fills above are safe. Direct fills
validate their complete source range before evaluating the mapper; captured
fills use the schedule's runtime-checked aggregate count. Fallible fills stop
at the first error while retaining a valid initialized prefix. Overflow,
overfill, underfill, and mismatched validity lengths must be checked in release
builds. A custom runtime may instead re-export the shared implementation when
it uses the same Polars identity.

```rust
mod runtime; // Implements the checked contract linked above.

use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::runtime::dataframe::ToDataFrame")]
struct Row {
    id: u32,
    values: Vec<String>,
}
```

The derive implements only `ColumnarSpec`. Keeping `Columnar`
blanket-provided makes the checked sink boundary unoverrideable for every
derived or manually specified encoder.

## Decimal Backends

`df-derive-core` provides `Decimal128Encode for rust_decimal::Decimal` behind
the `rust_decimal` feature, which is enabled by default on both `df-derive`
and `df-derive-core`.

To disable it:

```toml
df-derive = { version = "0.4", default-features = false }
```

Custom decimal backends should implement `Decimal128Encode` and use
`#[df_derive(decimal(precision = N, scale = S))]` on fields that should be
encoded as Polars decimal columns. Implementations must return an `i128`
mantissa rescaled to the requested scale, using round-half-to-even when
scaling down. Returning `None` surfaces as a Polars compute error. The
generated code verifies that the returned mantissa fits the declared precision
before constructing the Polars decimal column.

Unannotated decimal detection is syntax-based. A procedural macro receives
tokens, not rustc's resolved type information, so bare `Decimal` and canonical
`rust_decimal::Decimal` are treated as decimals automatically. Qualified paths
such as `domain::Decimal` are treated as nested custom structs unless you opt
them into decimal encoding with `decimal(...)`.

Temporal detection is syntax-based for the same reason. Bare or canonical
`chrono::NaiveDate`, `chrono::NaiveTime`, `chrono::NaiveDateTime`,
`chrono::DateTime<Tz>`, `chrono::Duration`, and `chrono::TimeDelta` are treated
as temporal types, along with `std::time::Duration` and
`core::time::Duration`. Qualified domain paths such as `domain::NaiveDate`
remain custom structs.

If your decimal trait lives somewhere other than the discovered runtime module,
point at it explicitly:

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(
    trait = "my_runtime::dataframe::ToDataFrame",
    decimal128_encode = "my_runtime::decimal_backend::Decimal128Encode",
)]
struct Tx {
    #[df_derive(decimal(precision = 38, scale = 10))]
    amount: MyDecimal,
}
```

## Compatibility

- **Rust edition**: 2024
- **Minimum supported Rust version**: 1.91. This is above the edition's
  1.85 floor because the Polars 0.54 dependency graph uses language features
  that first compile on Rust 1.91.
- **Polars**: 0.54
- **polars-arrow**: 0.54 through the default runtime facade. Custom runtimes
  selected with explicit trait overrides need a compatible direct dependency
  only for derived field shapes that emit public Arrow array builders; explicit
  facade/core runtime paths keep using the hidden default-runtime re-export.
- **Polars feature flags**: the default `df-derive` facade and
  `df-derive-core` runtime enable every Polars dtype flag required by the
  support matrix above. If you use `df-derive-macros` with a custom runtime
  and no `df-derive-core` dependency, enable the matching Polars feature
  flags on that runtime's `polars` dependency.

## Performance Notes

Using `df_derive::dataframe::Columnar` instead of `paft::dataframe::Columnar`
has no inherent runtime performance penalty. The macro generates the hot
column-building code at the impl site either way; the runtime path only
selects which trait receives the impl.

The public `Columnar::encode` boundary accepts slices,
`refs.iter().copied()`, and arbitrary one-shot iterators. It counts the rows
actually yielded and validates the generated `ColumnarSpec` output against its
explicit schema before constructing the outer frame. For slices and `Vec<T>`,
prefer `rows.as_slice().to_dataframe()`: the slice extension uses the known
row count directly. Use `T::encode(...)` when the input is a general iterator.

Generated code consumes the caller's source iterator once. Selected infallible
primitive-list shapes may capture segment references, or one reference per
penultimate nested-list group, and fill exact storage after that scan. Selected
wide scalar-tuple shapes may replay rows in bounded terminal lanes to keep hot
loops narrow. Slice conversion replays directly from the source slice;
generated nested encoders replay directly from the child-reference slice they
already collected. A general one-shot iterator captures yielded row references
only when its shape requests row replay. Fallible conversions and user-defined
work on direct fields remain in the source pass, so their errors do not consume
later source rows. Nested children are batch-encoded after their references are
collected: a child error stops later child evaluation, but the parent source
iterator has already been consumed.

The generated hot path is shape-dependent. Direct primitive scalar fields
normally share the source scan; wide safe tuple terminals may instead replay
in bounded lanes. Nested fields collect references and call the child's checked
batch encoder; the parent consumes validated columns directly, without
allocating a temporary child `DataFrame`. Tuple siblings also share source
resolution, list traversal, offsets, and validity before materializing their
individual columns.

Replay and deferred list schedules use temporary references rather than
cloning values. A replaying general iterator allocates one shared row-reference
vector proportional to the number of yielded rows. Slice-backed and nested
reference-slice cursors do not allocate that additional row buffer. Selected
primitive-list leaves may allocate reference vectors proportional to the
number of non-empty leaf segments or penultimate groups. Large one-shot
iterators should account for that temporary pointer storage.

Criterion benches in `df-derive/benches/` cover wide rows, nested structs,
deep Vec shapes, decimals, strings, borrowed data, tuple fields, and targeted
tuple-heavy / nested-heavy cost-model shapes.

Performance is continuously monitored with
[Bencher](https://bencher.dev/perf/df-derive).

## Examples

Run any example with:

```sh
cargo run -p df-derive --example quickstart
cargo run -p df-derive --example <example_name>
```

Available examples:

- **`quickstart`**: basic usage with single values and slices.
- **`nested`**: nested structs flattened with dot notation.
- **`vec_custom`**: `Vec<T>` fields and custom nested structs as list columns.
- **`tuple`**: tuple structs and `field_0`/`field_1` naming.
- **`datetime_decimal`**: chrono datetime values and `rust_decimal::Decimal`.
- **`as_string`**: `#[df_derive(as_string)]` for enums and custom values.
- **`generics`**: generic structs, default type parameters, and `()` payloads.
- **`nested_options`**: nested optional structs.
- **`deep_vec`**: deep `Vec<Vec<Vec<T>>>` list nesting.
- **`multi_option_vec`**: multiple `Option` layers above a `Vec`.
- **`nested_generics`**: generic structs used as nested fields and list items.

## License

MIT. See `LICENSE`.
