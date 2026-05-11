# df-derive

[![Crates.io](https://img.shields.io/crates/v/df-derive.svg)](https://crates.io/crates/df-derive)
[![Docs.rs](https://docs.rs/df-derive/badge.svg)](https://docs.rs/df-derive)
[![CI](https://github.com/gramistella/df-derive/actions/workflows/ci.yml/badge.svg)](https://github.com/gramistella/df-derive/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/crates/d/df-derive)](https://crates.io/crates/df-derive)
[![License](https://img.shields.io/crates/l/df-derive)](LICENSE)

Procedural derive macros for converting your Rust types into Polars `DataFrame`s.

## What this crate does

Deriving `ToDataFrame` on your structs and tuple structs generates fast, allocation-conscious code to:

- Convert a single value to a `polars::prelude::DataFrame`
- Convert a slice of values via a columnar path (efficient batch conversion)
- Inspect the schema (column names and `DataType`s) at compile time via a generated method

It supports nested structs (flattened with dot notation), `Option<T>`, `Vec<T>`, tuple structs, and key domain types like `chrono::DateTime<Tz>`, `chrono::NaiveDateTime`, and decimal backend paths named `Decimal`.

## Installation

Add the macro crate, Polars, and `polars-arrow`. You will also need a trait defining the `to_dataframe` behavior (you can use your own runtime crate/traits; see the override section below). For a minimal inline trait you can copy, see the Quick start example.

```toml
[dependencies]
df-derive = "0.3.0"
polars = { version = "0.53", features = ["timezones", "dtype-date", "dtype-time", "dtype-duration", "dtype-decimal"] }
polars-arrow = "0.53"

# If you use these types in your models
chrono = { version = "0.4", features = ["serde"] }
rust_decimal = { version = "1.36", features = ["serde"] }
```

`df-derive` requires `polars-arrow` as a direct dependency (it is compiled transitively via `polars` but not re-exported through `polars::prelude`). Add `polars-arrow = "0.53"` to your `Cargo.toml` alongside `polars`.

## Quick start

Copy-paste runnable example without any external runtime traits. This is a complete working example that you can run with `cargo run --example quickstart`.

Cargo.toml:

```toml
[package]
name = "quickstart"
version = "0.1.0"
edition = "2024"

[dependencies]
df-derive = "0.3"
polars = { version = "0.53", features = ["timezones", "dtype-date", "dtype-time", "dtype-duration", "dtype-decimal"] }
polars-arrow = "0.53"
```

src/main.rs:

```rust
use df_derive::ToDataFrame;

mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
    }

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let refs: Vec<&Self> = items.iter().collect();
            Self::columnar_from_refs(&refs)
        }
        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
    }

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar + ToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as ToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }
}

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")] // Columnar path auto-infers to crate::dataframe::Columnar
struct Trade {
    symbol: String,
    price: f64,
    size: u64,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let t = Trade { symbol: "AAPL".into(), price: 187.23, size: 100 };
    let df_single = <Trade as crate::dataframe::ToDataFrame>::to_dataframe(&t)?;
    println!("{}", df_single);

    let trades = vec![
        Trade { symbol: "AAPL".into(), price: 187.23, size: 100 },
        Trade { symbol: "MSFT".into(), price: 411.61, size: 200 },
    ];
    use crate::dataframe::ToDataFrameVec;
    let df_batch = trades.as_slice().to_dataframe()?;
    println!("{}", df_batch);

    Ok(())
}
```

Run it:

```bash
cargo run
```

> **Skip the boilerplate.** The `dataframe` module above is the same on every project that doesn't already use a `paft` runtime. The sibling crate `df-derive-runtime` ships it (and the reference `Decimal128Encode for rust_decimal::Decimal` impl) so you don't have to inline it yourself. Add `df-derive-runtime = "0.3"` to your `[dependencies]` and replace the `#[df_derive(...)]` attribute's trait path with `df_derive_runtime::dataframe::ToDataFrame`. See the [Using `df-derive-runtime`](#using-df-derive-runtime) section below for the full snippet.

## Features

- **Nested structs (flattening)**: fields of nested structs appear as `outer.inner` columns
- **Vec of primitives and structs**: becomes Polars `List` columns; `Vec<Nested>` becomes multiple `outer.subfield` list columns
- **`Option<T>`**: null-aware materialization for both scalars and lists
- **Tuple structs**: supported; columns are named `field_0`, `field_1`, ...
- **Tuple-typed fields**: `pair: (A, B)` flattens to `pair.field_0`, `pair.field_1`. Outer
  `Option`/`Vec` distribute across element columns: `Vec<(A, B)>` produces parallel `List`
  columns. Nested tuples work when the outer tuple is not itself wrapped; use a named struct
  for nested tuple shapes inside an outer `Option` or `Vec`. Smart-pointer composition works.
  Unit `()` is rejected. Field-level attributes don't apply to tuple fields (hoist into a
  named struct instead).
- **Empty structs**: produce `(1, 0)` for instances and `(0, 0)` for empty frames
- **Schema discovery**: `T::schema() -> Vec<(String, DataType)>`
- **Columnar batch conversion**: `[T]::to_dataframe()` via the `Columnar` implementation

### Attribute helpers

Use `#[df_derive(as_string)]` to stringify values during conversion. This is particularly useful for enums:

```rust
#[derive(Clone, Debug, PartialEq)]
enum Status { Active, Inactive }

// Required: implement Display for the enum
impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Active => write!(f, "Active"),
            Status::Inactive => write!(f, "Inactive"),
        }
    }
}

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct WithEnums {
    #[df_derive(as_string)]
    status: Status,
    #[df_derive(as_string)]
    opt_status: Option<Status>,
    #[df_derive(as_string)]
    statuses: Vec<Status>,
}
```

Columns will use `DataType::String` (or `List<String>` for `Vec<_>`), and values are produced via `ToString`. See the complete working example with `cargo run --example as_string`.

#### Borrowing strings: `#[df_derive(as_str)]`

If your field's type already implements `AsRef<str>` — validated newtypes, interned identifiers, enums with a `match`-based string view — prefer `#[df_derive(as_str)]`. It populates the same `Vec<&str>` / `Vec<Option<&str>>` columnar buffer the bare-`String` borrowing path uses, so the per-row `String` allocation that `as_string` performs is avoided entirely.

```rust
#[derive(Clone, Debug, PartialEq)]
enum Status { Active, Inactive }

impl AsRef<str> for Status {
    fn as_ref(&self) -> &str {
        match self {
            Status::Active => "ACTIVE",
            Status::Inactive => "INACTIVE",
        }
    }
}

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct WithEnums {
    #[df_derive(as_str)] status: Status,
    #[df_derive(as_str)] opt_status: Option<Status>,
    #[df_derive(as_str)] statuses: Vec<Status>,
    #[df_derive(as_str)] opt_statuses: Option<Vec<Status>>,
}
```

Selection rule: if your field's type implements `AsRef<str>`, prefer `as_str` — it borrows the string for the duration of the conversion. Otherwise use `as_string`, which formats via `Display` and allocates a `String` per row. The two attributes are mutually exclusive on a given field; using both raises a compile error pointing at the field.

`Vec<T>` and `Option<Vec<T>>` shapes also benefit — the inner `Series` is built from a borrowed `&str` iterator, so there is no per-element clone either.

#### Byte blobs: `#[df_derive(as_binary)]`

For fields representing opaque byte blobs, opt into a Polars `Binary` column with `#[df_derive(as_binary)]`. Without the attribute, `Vec<u8>` continues to materialize as `List(UInt8)`, and borrowed byte slices (`&[u8]` / `Cow<'_, [u8]>`) are rejected with a hint to add the attribute; the attribute is the single decision point for choosing the byte-blob representation.

```rust
#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Record {
    #[df_derive(as_binary)] payload: Vec<u8>,                 // Binary
    #[df_derive(as_binary)] maybe_payload: Option<Vec<u8>>,   // Binary (nullable)
    #[df_derive(as_binary)] payloads: Vec<Vec<u8>>,           // List(Binary)
    #[df_derive(as_binary)] sparse: Vec<Option<Vec<u8>>>,     // List(Binary), per-element nullable
    #[df_derive(as_binary)] outer_opt: Option<Vec<Vec<u8>>>,  // List(Binary), nullable outer
    #[df_derive(as_binary)] borrowed_slice: &'static [u8],    // Binary
    #[df_derive(as_binary)] borrowed: std::borrow::Cow<'static, [u8]>, // Binary
}
```

Accepted shapes: `Vec<u8>`, `Option<Vec<u8>>`, `Vec<Vec<u8>>`, `Vec<Option<Vec<u8>>>`, `Option<Vec<Vec<u8>>>`, plus the same scalar/list/nullable-list shapes over `&[u8]` and `Cow<'_, [u8]>`. Rejected at compile time: bare `u8`, `Option<u8>`, `Vec<Option<u8>>` (BinaryView cannot carry per-byte nulls), and any non-`u8` leaf (e.g. `Vec<i32>`, `&[i32]`, `Cow<'_, [i32]>`, `String`). The attribute is mutually exclusive with `as_str`, `as_string`, `decimal(...)`, and `time_unit = "..."`.

## Supported types

- **Primitives**: `String`, `&str`, `bool`, integer types (`i8/i16/i32/i64/i128/isize`, `u8/u16/u32/u64/u128/usize`), `std::num::NonZero*` integer types, `f32`, `f64`
- **Time**: `chrono::DateTime<Tz>` and `chrono::NaiveDateTime` → `Datetime(Milliseconds, None)` by default; override with `#[df_derive(time_unit = "ms"|"us"|"ns")]`. `DateTime<Tz>` values are encoded as UTC instants; the textual timezone/offset is not preserved, so use `#[df_derive(as_string)]` if that representation matters.
- **Date / time-of-day**: `chrono::NaiveDate` → `Date` (i32 days since 1970-01-01; requires Polars `dtype-date`), `chrono::NaiveTime` → `Time` (i64 ns since midnight; requires Polars `dtype-time`). Both have fixed encodings — `time_unit` is not accepted.
- **Duration**: `std::time::Duration`, `core::time::Duration`, and `chrono::Duration` (alias for `chrono::TimeDelta`) → `Duration(Nanoseconds)` by default (requires Polars `dtype-duration`); override with `#[df_derive(time_unit = "ms"|"us"|"ns")]`. Bare `Duration` (no qualifier) is rejected as ambiguous — write `std::time::Duration`, `core::time::Duration`, or `chrono::Duration`.
- **Decimal**: any type path whose last segment is `Decimal` (for example
  `rust_decimal::Decimal` or a backend facade such as `paft_decimal::Decimal`)
  → `Decimal(38, 10)`. This implicit detection is syntax-based because proc
  macros cannot resolve type aliases. For differently named decimal backends,
  use `#[df_derive(decimal(precision = N, scale = S))]` and implement
  `Decimal128Encode`.
- **Binary blobs**: opt-in per field with `#[df_derive(as_binary)]` over a `Vec<u8>`, `&[u8]`, or `Cow<'_, [u8]>` shape; default `Vec<u8>` (no attribute) remains `List(UInt8)`, while unannotated borrowed byte slices are rejected
- **Wrappers**: `Option<T>`, `Vec<T>` in any nesting order
- **Transparent pointers**: `Box<T>`, `Rc<T>`, `Arc<T>`, borrowed references `&T`, and `Cow<'_, T>` (with sized inner) peel transparently — column shape, schema dtype, and runtime are identical to the bare `T` field. Composes freely with `Option`/`Vec` (e.g. `Option<&i32>`, `Vec<Arc<String>>`, `Box<Vec<f64>>`). `&str` and `Cow<'_, str>` are treated as borrowed string leaves by default. `&[u8]` and `Cow<'_, [u8]>` are supported with `#[df_derive(as_binary)]`; other borrowed slice forms are rejected — use `Vec<T>` for list columns.
- **Custom structs**: any other struct deriving `ToDataFrame` (supports nesting and `Vec<Nested>`)
- **Tuple structs**: unnamed fields are emitted as `field_{index}`
- **Tuple-typed fields**: tuples like `(A, B)`, `Option<(A, B)>`, `Vec<(A, B)>`, and unwrapped nested `((A, B), C)` flatten to one column per element with `<field>.field_<i>` names. The outer wrapper distributes across every element column for non-nested tuples. Nested tuples inside an outer `Option` or `Vec` are rejected; hoist the inner tuple into a named struct. Unit `()` and field-level attributes (`as_str`, `decimal`, `time_unit`, …) are rejected on tuple fields.

## Column naming

- Named struct fields: `field_name`
- Nested structs: `outer.inner` (recursively)
- Vec of custom structs: `vec_field.subfield` (list dtype)
- Tuple-typed fields: `field.field_0`, `field.field_1` (recursively for nested tuples)
- Tuple structs: `field_0`, `field_1`, ...

## Generated API

For every `#[derive(ToDataFrame)]` type `T` the macro generates implementations of two traits (paths configurable via `#[df_derive(...)]`):

- `ToDataFrame` for `T`:
  - `fn to_dataframe(&self) -> PolarsResult<DataFrame>`
  - `fn empty_dataframe() -> PolarsResult<DataFrame>`
  - `fn schema() -> PolarsResult<Vec<(String, DataType)>>`
- `Columnar` for `T`:
  - `fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>`
  - `fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>` is expected as a trait default that delegates to `columnar_from_refs`

## Examples

This crate includes several runnable examples in the `examples/` directory. You can run any example with:

```bash
cargo run --example <example_name>
```

Or run all examples to see the full feature set:

```bash
cargo run --example quickstart && \
cargo run --example nested && \
cargo run --example vec_custom && \
cargo run --example tuple && \
cargo run --example datetime_decimal && \
cargo run --example as_string
```

### Available Examples

- **`quickstart`** - Basic usage with single and batch DataFrame conversion
- **`nested`** - Nested structs with dot notation column naming  
- **`vec_custom`** - Vec of custom structs creating List columns
- **`tuple`** - Tuple structs with field_0, field_1 naming
- **`datetime_decimal`** - DateTime and Decimal type support
- **`as_string`** - `#[df_derive(as_string)]` attribute for enum conversion

### Example Code Snippets

#### Nested structs

```rust
#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Address { street: String, city: String, zip: u32 }

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Person { name: String, age: u32, address: Address }

// Columns: name, age, address.street, address.city, address.zip
```

> Note: the runnable examples define a small `dataframe` module with the traits used by the macro. Some helper trait items are not used in every snippet (for example `empty_dataframe` or `Columnar`). To avoid noise during `cargo run --example …`, the examples annotate that module with `#[allow(dead_code)]`.

#### Vec of custom structs

```rust
#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Quote { ts: i64, open: f64, high: f64, low: f64, close: f64, volume: u64 }

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct MarketData { symbol: String, quotes: Vec<Quote> }

// Columns include: symbol, quotes.ts, quotes.open, quotes.high, ... (each a List)
```

#### Tuple structs

```rust
#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct SimpleTuple(i32, String, f64);
// Columns: field_0 (Int32), field_1 (String), field_2 (Float64)
```

#### DateTime and Decimal

```rust
#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct TxRecord { amount: rust_decimal::Decimal, ts: chrono::DateTime<chrono::Utc> }
// Schema dtypes: amount = Decimal(38, 10), ts = Datetime(Milliseconds, None)
```

> Why `#[allow(dead_code)]` in examples? The examples include a minimal `dataframe` module to provide the traits that the macro implements. Not every example calls every method (e.g., `empty_dataframe`, `schema`), and compile-time warnings would otherwise distract from the output. Adding `#[allow(dead_code)]` to that module keeps the examples clean while remaining fully correct.

#### As string attribute

```rust
#[derive(Clone, Debug, PartialEq)]
enum Status { Active, Inactive }

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Active => write!(f, "Active"),
            Status::Inactive => write!(f, "Inactive"),
        }
    }
}

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct WithEnums {
    #[df_derive(as_string)]
    status: Status,
    #[df_derive(as_string)]
    opt_status: Option<Status>,
    #[df_derive(as_string)]
    statuses: Vec<Status>,
}
// Columns use DataType::String or List<String>
```

> **Note**: All examples require the trait definitions shown in the Quick start section. See the complete working examples in the `examples/` directory.

## Limitations and guidance

- **Unsupported container types**: maps/sets like `HashMap<_, _>` are not supported. The rejection error suggests `Vec<(K, V)>` as a workaround — that conversion now works directly (tuple-typed fields are supported).
- **Enums**: derive on enums is not supported; use `#[df_derive(as_string)]` on enum fields.
- **Generics**: generic structs are supported. The macro injects `ToDataFrame + Columnar` bounds on every type parameter, plus `Decimal128Encode` for generic parameters explicitly annotated with `decimal(...)`. The unit type `()` can be used as a payload to contribute zero columns.
- **All nested types must also derive**: if you nest a struct, it must also derive `ToDataFrame`.

## Performance notes

- The derive implements an internal `Columnar` path used by the runtime to convert slices efficiently, avoiding per-row DataFrame builds.
- Criterion benches in `benches/` exercise wide, deep, and nested-Vec shapes (100k+ rows), demonstrating consistent performance across shapes.

### Performance tracking

Performance is continuously monitored and tracked using [Bencher](https://bencher.dev):

<a href="https://bencher.dev/perf/df-derive?key=true&reports_per_page=4&branches_per_page=8&testbeds_per_page=8&benchmarks_per_page=8&plots_per_page=8&reports_page=1&branches_page=1&testbeds_page=1&benchmarks_page=1&plots_page=1&report=4c6575f5-d966-4253-a96c-fdfb648b519b&branches=9c5e70f8-d048-4fa3-9934-3b7c065c942b&heads=f8bc9f79-a393-4bc7-8235-55abe5c7cd6a&testbeds=227a10de-e9f3-44eb-853f-9077a4b00383&benchmarks=a474ffbe-58f9-481d-acdc-c5d918dadf77%2Cafded7d7-93d7-4e29-897d-3f458e7c9b1b%2C59daa624-685a-44f9-824b-5bff19e13724%2Cd45cb2ac-af9f-4a7c-818c-a56a244119a4%2Ce5cca53e-8583-4b07-8eee-59ef09afa37b%2C00f9a237-00ad-4eb8-a867-4c7f7986ba28%2C590d7f1f-ccad-4c23-9d3d-d5aa835f1529%2C99a2763d-53e0-426b-bd78-0bafa18be4e1&measures=23d46ce3-69ec-4fb4-936d-b6f294f0861a&start_time=1755304749000&end_time=1789430400000&lower_boundary=false&upper_boundary=false&clear=true&x_axis=date_time&utm_medium=share&utm_source=bencher&utm_content=img&utm_campaign=perf%2Bimg&utm_term=df-derive"><img src="https://api.bencher.dev/v0/projects/df-derive/perf/img?branches=9c5e70f8-d048-4fa3-9934-3b7c065c942b&heads=f8bc9f79-a393-4bc7-8235-55abe5c7cd6a&testbeds=227a10de-e9f3-44eb-853f-9077a4b00383&benchmarks=a474ffbe-58f9-481d-acdc-c5d918dadf77%2Cafded7d7-93d7-4e29-897d-3f458e7c9b1b%2C59daa624-685a-44f9-824b-5bff19e13724%2Cd45cb2ac-af9f-4a7c-818c-a56a244119a4%2Ce5cca53e-8583-4b07-8eee-59ef09afa37b%2C00f9a237-00ad-4eb8-a867-4c7f7986ba28%2C590d7f1f-ccad-4c23-9d3d-d5aa835f1529%2C99a2763d-53e0-426b-bd78-0bafa18be4e1&measures=23d46ce3-69ec-4fb4-936d-b6f294f0861a&start_time=1755304749000&end_time=1789430400000" title="df-derive" alt="df-derive - Bencher" /></a>

## Compatibility

- **Rust edition**: 2024
- **Polars**: 0.53 (tested)
- **polars-arrow**: 0.53 (direct dependency required by generated code)
- Enable Polars features `timezones` for timezone-aware `DateTime<Tz>`,
  `dtype-date` for `NaiveDate`, `dtype-time` for `NaiveTime`,
  `dtype-duration` for duration columns, `dtype-i8` / `dtype-i16` /
  `dtype-u8` / `dtype-u16` for exact small-integer columns,
  `dtype-i128` / `dtype-u128` for 128-bit integer columns, and
  `dtype-decimal` for `Decimal`.

## License

MIT. See `LICENSE`.

## Crate path override (about paft)

This crate currently resolves default trait paths to a `dataframe` module under the `paft` ecosystem. Concretely, it attempts to implement:

- `paft::dataframe::ToDataFrame`, `paft::dataframe::Columnar`, and `paft::dataframe::Decimal128Encode` if the `paft` facade is present.
- `paft_utils::dataframe::ToDataFrame`, `paft_utils::dataframe::Columnar`, and `paft_utils::dataframe::Decimal128Encode` if `paft-utils` is present without the facade.
- `crate::core::dataframe::...` as a local fallback for projects that keep their dataframe traits in a `core::dataframe` module and do not use paft.

You can override these paths for any runtime by annotating your type with `#[df_derive(...)]`:

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(trait = "my_runtime::dataframe::ToDataFrame")]
// `Columnar` and `Decimal128Encode` are inferred as
// `my_runtime::dataframe::Columnar` / `my_runtime::dataframe::Decimal128Encode`
struct MyType { /* fields */ }
```

If you need to override them explicitly:

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(
    trait = "my_runtime::dataframe::ToDataFrame",
    columnar = "my_runtime::dataframe::Columnar",
    decimal128_encode = "my_runtime::dataframe::Decimal128Encode",
)]
struct MyType { /* fields */ }
```

### Using `df-derive-runtime`

If you don't already have a `paft` runtime, depend on `df-derive-runtime` instead of inlining the trait module yourself.

The derive macro only generates impls; it does not own the traits those impls target. That is deliberate, because paft users already have a `paft::dataframe` trait module. For everyone else, `df-derive-runtime` is the small canonical trait module: it ships `ToDataFrame`, `Columnar`, `ToDataFrameVec`, `Decimal128Encode`, the `()` impls used by generic `Wrapper<()>` shapes, and the reference `Decimal128Encode for rust_decimal::Decimal` impl (gated behind the `rust_decimal` feature, which is enabled by default).

```toml
[dependencies]
df-derive = "0.3"
polars = { version = "0.53", features = ["timezones", "dtype-date", "dtype-time", "dtype-duration", "dtype-decimal"] }
polars-arrow = "0.53"
df-derive-runtime = "0.3"
```

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
struct MyType { /* fields */ }
```

The `Columnar` and `Decimal128Encode` paths are auto-inferred from the `trait` attribute, so the single `trait = "..."` line is all you need. Since the runtime crate ships the reference decimal impl, `Decimal` fields just work without any extra `impl` block in your code. Most of the runnable examples in this repository (everything except `quickstart.rs`) use this shape — see them for end-to-end snippets.

If you don't need the `rust_decimal` reference impl (e.g. you ship your own `Decimal128Encode for MyDecimal` and don't pull in `rust_decimal`), turn the feature off:

```toml
df-derive-runtime = { version = "0.3", default-features = false }
```

## Custom decimal backends

`Decimal` fields are converted to a polars `Decimal(precision, scale)` column by going through a small user-pluggable trait, `Decimal128Encode`:

```rust
pub trait Decimal128Encode {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
}
```

The implementer rescales the value to the schema scale and returns the mantissa as an `i128`. A `None` return surfaces as a polars `ComputeError` from the generated code, matching the existing scale-up overflow path.

**Implicit detection is name-based.** A procedural macro receives tokens, not
rustc's resolved type information, so unannotated decimal fields are detected
by the last path segment: `Decimal`, `rust_decimal::Decimal`,
`paft_decimal::Decimal`, or any re-export/type alias whose written path ends in
`Decimal`. This is intentional and keeps facade crates ergonomic, but it is a
heuristic.

**Explicit opt-in for custom names.** If your backend type is not written with a
final `Decimal` segment, add `#[df_derive(decimal(precision = N, scale = S))]`.
That attribute is a semantic assertion that the leaf should use the decimal
encoder; the generated code then calls the configured `Decimal128Encode` trait
and normal Rust type checking verifies that an impl exists.

**Rounding contract (load-bearing).** Implementations MUST use round-half-to-even (banker's rounding) on scale-down. Polars' own `str_to_dec128` rounds that way, so backends that disagree on tie-breaking (e.g., `rust_decimal::Decimal::rescale` rounds half-away-from-zero) would produce different bytes than the historical `to_string + cast` path the codegen replaced. Sticking to round-half-to-even keeps the column byte-identical across decimal backends.

**Default discovery.** The codegen looks for `Decimal128Encode` next to the `ToDataFrame` and `Columnar` traits — by default `paft::dataframe::Decimal128Encode`, then `paft_utils::dataframe::Decimal128Encode`, then `crate::core::dataframe::Decimal128Encode` as a non-paft local fallback. If you point `trait = "..."` at your own `ToDataFrame`, the `Columnar` and `Decimal128Encode` paths are inferred by replacing the last path segment. In the common facade case, expose the backend as `Decimal` and add an `impl Decimal128Encode for Decimal`; if the backend is written as `MyDecimal`, add the field-level `decimal(...)` attribute too.

**Per-derive override.** If your decimal trait lives somewhere else, point at it explicitly:

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(
    trait = "my_runtime::dataframe::ToDataFrame",
    decimal128_encode = "my_runtime::decimal_backend::Decimal128Encode",
)]
struct Tx { /* … */ }
```

**Backend status.** Facade crates can expose their active backend as a type named
`Decimal` for implicit support, or users can annotate differently named backend
fields with `decimal(...)`. The sibling crate `df-derive-runtime` ships the
reference `Decimal128Encode for rust_decimal::Decimal` impl behind a default-on
`rust_decimal` feature — depend on it (see [Using `df-derive-runtime`](#using-df-derive-runtime) above) and the codegen finds the impl via the auto-inferred `df_derive_runtime::dataframe::Decimal128Encode` path. A copy-paste-ready inline form also lives in `tests/common.rs` for projects that prefer not to take the dependency.

The repository also contains an unpublished `df-derive-test-harness` workspace crate that validates the reference decimal implementation against Polars. It is intentionally not part of the public release surface for v0.3.0.
