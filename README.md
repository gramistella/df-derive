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

It supports nested structs (flattened with dot notation), `Option<T>`, `Vec<T>`, tuple structs, and key domain types like `chrono::DateTime<Utc>` and `rust_decimal::Decimal`.

## Installation

Add the macro crate and Polars. You will also need a trait defining the `to_dataframe` behavior (you can use your own runtime crate/traits; see the override section below). For a minimal inline trait you can copy, see the Quick start example.

```toml
[dependencies]
df-derive = "0.1"
polars = { version = "0.50", features = ["timezones", "dtype-decimal"] }

# If you use these types in your models
chrono = { version = "0.4", features = ["serde"] }
rust_decimal = { version = "1.36", features = ["serde"] }
```

## Quick start

Copy-paste runnable example without any external runtime traits. This is a complete working example that you can run with `cargo run --example quickstart`.

Cargo.toml:

```toml
[package]
name = "quickstart"
version = "0.1.1"
edition = "2024"

[dependencies]
df-derive = "0.1"
polars = { version = "0.50", features = ["timezones", "dtype-decimal"] }
```

src/main.rs:

```rust
use df_derive::ToDataFrame;

mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
    }

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
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

## Features

- **Nested structs (flattening)**: fields of nested structs appear as `outer.inner` columns
- **Vec of primitives and structs**: becomes Polars `List` columns; `Vec<Nested>` becomes multiple `outer.subfield` list columns
- **`Option<T>`**: null-aware materialization for both scalars and lists
- **Tuple structs**: supported; columns are named `field_0`, `field_1`, ...
- **Empty structs**: produce `(1, 0)` for instances and `(0, 0)` for empty frames
- **Schema discovery**: `T::schema() -> Vec<(&'static str, DataType)>`
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

## Supported types

- **Primitives**: `String`, `bool`, integer types (`i8/i16/i32/i64/isize`, `u8/u16/u32/u64/usize`), `f32`, `f64`
- **Time**: `chrono::DateTime<Utc>` → materialized as `Datetime(Milliseconds, None)`
- **Decimal**: `rust_decimal::Decimal` → `Decimal(38, 10)`
- **Wrappers**: `Option<T>`, `Vec<T>` in any nesting order
- **Custom structs**: any other struct deriving `ToDataFrame` (supports nesting and `Vec<Nested>`)
- **Tuple structs**: unnamed fields are emitted as `field_{index}`

## Column naming

- Named struct fields: `field_name`
- Nested structs: `outer.inner` (recursively)
- Vec of custom structs: `vec_field.subfield` (list dtype)
- Tuple structs: `field_0`, `field_1`, ...

## Generated API

For every `#[derive(ToDataFrame)]` type `T` the macro generates implementations of two traits (paths configurable via `#[df_derive(...)]`):

- `ToDataFrame` for `T`:
  - `fn to_dataframe(&self) -> PolarsResult<DataFrame>`
  - `fn empty_dataframe() -> PolarsResult<DataFrame>`
  - `fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>`
- `Columnar` for `T`:
  - `fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>`

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

#### `DateTime<Utc>` and Decimal

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

- **Unsupported container types**: maps/sets like `HashMap<_, _>` are not supported.
- **Enums**: derive on enums is not supported; use `#[df_derive(as_string)]` on enum fields.
- **Generics**: generic structs are not supported by the derive (see tests/fail for examples).
- **All nested types must also derive**: if you nest a struct, it must also derive `ToDataFrame`.

## Performance notes

- The derive implements an internal `Columnar` path used by the runtime to convert slices efficiently, avoiding per-row DataFrame builds.
- Criterion benches in `benches/` exercise wide, deep, and nested-Vec shapes (100k+ rows), demonstrating consistent performance across shapes.

### Performance tracking

Performance is continuously monitored and tracked using [Bencher](https://bencher.dev):

<a href="https://bencher.dev/perf/df-derive?key=true&reports_per_page=4&branches_per_page=8&testbeds_per_page=8&benchmarks_per_page=8&plots_per_page=8&reports_page=1&branches_page=1&testbeds_page=1&benchmarks_page=1&plots_page=1&report=4c6575f5-d966-4253-a96c-fdfb648b519b&branches=9c5e70f8-d048-4fa3-9934-3b7c065c942b&heads=f8bc9f79-a393-4bc7-8235-55abe5c7cd6a&testbeds=227a10de-e9f3-44eb-853f-9077a4b00383&benchmarks=a474ffbe-58f9-481d-acdc-c5d918dadf77%2Cafded7d7-93d7-4e29-897d-3f458e7c9b1b%2C59daa624-685a-44f9-824b-5bff19e13724%2Cd45cb2ac-af9f-4a7c-818c-a56a244119a4%2Ce5cca53e-8583-4b07-8eee-59ef09afa37b%2C00f9a237-00ad-4eb8-a867-4c7f7986ba28%2C590d7f1f-ccad-4c23-9d3d-d5aa835f1529%2C99a2763d-53e0-426b-bd78-0bafa18be4e1&measures=23d46ce3-69ec-4fb4-936d-b6f294f0861a&start_time=1755304749000&end_time=1789430400000&lower_boundary=false&upper_boundary=false&clear=true&x_axis=date_time&utm_medium=share&utm_source=bencher&utm_content=img&utm_campaign=perf%2Bimg&utm_term=df-derive"><img src="https://api.bencher.dev/v0/projects/df-derive/perf/img?branches=9c5e70f8-d048-4fa3-9934-3b7c065c942b&heads=f8bc9f79-a393-4bc7-8235-55abe5c7cd6a&testbeds=227a10de-e9f3-44eb-853f-9077a4b00383&benchmarks=a474ffbe-58f9-481d-acdc-c5d918dadf77%2Cafded7d7-93d7-4e29-897d-3f458e7c9b1b%2C59daa624-685a-44f9-824b-5bff19e13724%2Cd45cb2ac-af9f-4a7c-818c-a56a244119a4%2Ce5cca53e-8583-4b07-8eee-59ef09afa37b%2C00f9a237-00ad-4eb8-a867-4c7f7986ba28%2C590d7f1f-ccad-4c23-9d3d-d5aa835f1529%2C99a2763d-53e0-426b-bd78-0bafa18be4e1&measures=23d46ce3-69ec-4fb4-936d-b6f294f0861a&start_time=1755304749000&end_time=1789430400000" title="df-derive" alt="df-derive - Bencher" /></a>

## Compatibility

- **Rust edition**: 2024
- **Polars**: 0.50 (tested)
- Enable Polars features `timezones` and `dtype-decimal` if you use `DateTime<Utc>` or `Decimal`.

## License

MIT. See `LICENSE`.

## Crate path override (about paft)

This crate currently resolves default trait paths to a `dataframe` module under the `paft` ecosystem. Concretely, it attempts to implement:

- `paft::dataframe::ToDataFrame` and `paft::dataframe::Columnar` (or `paft-core::dataframe::...`) if those crates are present.

You can override these paths for any runtime by annotating your type with `#[df_derive(...)]`:

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(trait = "my_runtime::dataframe::ToDataFrame")] // Columnar will be inferred as my_runtime::dataframe::Columnar
struct MyType { /* fields */ }
```

If you need to override both explicitly:

```rust
#[derive(df_derive::ToDataFrame)]
#[df_derive(
    trait = "my_runtime::dataframe::ToDataFrame",
    columnar = "my_runtime::dataframe::Columnar",
)]
struct MyType { /* fields */ }
```
