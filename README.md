# df-derive

[![Crates.io](https://img.shields.io/crates/v/df-derive.svg)](https://crates.io/crates/df-derive)
[![Docs.rs](https://docs.rs/df-derive/badge.svg)](https://docs.rs/df-derive)
[![CI](https://github.com/gramistella/df-derive/actions/workflows/ci.yml/badge.svg)](https://github.com/gramistella/df-derive/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/crates/d/df-derive)](https://crates.io/crates/df-derive)
[![License](https://img.shields.io/crates/l/df-derive)](LICENSE)

`df-derive` derives fast conversions from Rust structs into Polars
`DataFrame`s. Put `#[derive(ToDataFrame)]` on a struct and its values become
rows: one value, a slice, or any iterator of references.

## Quick Start

```toml
[dependencies]
df-derive = "0.6"
polars = "0.55"
```

```rust
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Trade {
    symbol: String,
    price: f64,
    size: Option<u64>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![
        Trade { symbol: "AAPL".into(), price: 187.23, size: Some(100) },
        Trade { symbol: "MSFT".into(), price: 411.61, size: None },
    ];

    let df = rows.as_slice().to_dataframe()?;
    println!("{df}");
    Ok(())
}
```

Generated code reaches Polars through `df_derive`'s own re-export, so you only
need a direct `polars` dependency for the Polars types your own code names.
The runtime enables the Polars dtype features in the table below.

## What You Get

For a derived type `T`, through the `ToDataFrame` trait:

| Call | Result |
| --- | --- |
| `value.to_dataframe()` | a one-row `DataFrame` |
| `slice.to_dataframe()` (also on `Vec<T>`) | one row per element |
| `T::rows_to_dataframe(iter)` | one row per `&T` the iterator yields |
| `T::empty_dataframe()` | zero rows, full schema |
| `T::schema()` | column names and dtypes, in order |

Every conversion writes each field straight into its column buffer, with no
intermediate frames or per-field copies of the rows.

## How It Works

The derive does not analyse your field types. Each supported type implements
the `Columnar` trait, which names a *builder* that owns the buffers for that
type's columns. A derived struct gets a builder with one member per field, and
wrappers compose by wrapping the builder of what they contain:

- `Option<T>` adds a validity bitmap around `T`'s builder.
- `Vec<T>` adds list offsets around `T`'s builder.
- A nested struct's builder is simply a member of its parent's builder.

Because the compiler resolves this through trait impls, type aliases, generic
parameters and types from other crates all work the same way as the type they
stand for.

## Supported Types

Any of these works as a field with no attribute, in any nesting:

- **Numbers and booleans**: `bool`, `i8`–`i128`, `u8`–`u128`, `isize`,
  `usize`, `f32`, `f64`, and the `std::num::NonZero*` integers.
- **Strings**: `String`, `&str`, `Box<str>`, `Cow<'_, str>`.
- **Dates and times** (`chrono` feature, on by default): `DateTime<Tz>` and
  `NaiveDateTime` as `Datetime(Milliseconds)`, `NaiveDate` as `Date`,
  `NaiveTime` as `Time`, `chrono::Duration` / `TimeDelta` as
  `Duration(Nanoseconds)`. `std::time::Duration` is `Duration(Nanoseconds)`
  with or without the feature.
- **Decimals** (`rust_decimal` feature, on by default):
  `rust_decimal::Decimal` as `Decimal(38, 10)`.
- **Other derived structs**: flattened into the parent with dot notation.
- **Tuples**: `(A, B)` becomes `field.field_0`, `field.field_1`.
- **`Option<T>`**: the same columns as `T`, nullable.
- **`Vec<T>`, `&[T]`, `Box<[T]>`**: the same columns as `T`, each wrapped in a
  `List`. A list of structs becomes one list column per struct field.
- **`HashMap<K, V>`, `BTreeMap<K, V>`**: two list columns, `field.key` and
  `field.value`.
- **Pointers**: `Box<T>`, `Rc<T>`, `Arc<T>`, `&T` and `Cow<'_, T>` encode as
  `T`.
- **`()`**: no columns. Useful as a generic payload.

Named structs, tuple structs (`field_0`, `field_1`, …), unit structs and
generic structs can all derive `ToDataFrame`.

| Rust type | Polars dtype | Polars feature |
| --- | --- | --- |
| `bool` | `Boolean` | none |
| `String`, `&str`, `as_str`, `as_string` | `String` | none |
| `i8`, `i16`, `u8`, `u16` | `Int8`, `Int16`, `UInt8`, `UInt16` | `dtype-i8`, `dtype-i16`, `dtype-u8`, `dtype-u16` |
| `i32`, `i64`, `isize`, `u32`, `u64`, `usize` | `Int32`, `Int64`, `UInt32`, `UInt64` | none |
| `i128`, `u128` | `Int128`, `UInt128` | `dtype-i128`, `dtype-u128` |
| `f32`, `f64` | `Float32`, `Float64` | none |
| `DateTime<Tz>`, `NaiveDateTime` | `Datetime` | `dtype-datetime` |
| `NaiveDate` | `Date` | `dtype-date` |
| `NaiveTime` | `Time` | `dtype-time` |
| `std::time::Duration`, `chrono::Duration` | `Duration` | `dtype-duration` |
| `rust_decimal::Decimal`, `decimal(...)` | `Decimal` | `dtype-decimal` |
| `as_binary` | `Binary` | none |

`isize` and `usize` are stored as 64-bit columns on every platform, so a
schema does not depend on the build target. `NonZero*` types use the dtype of
their base integer.

## Field Attributes

```rust
#[derive(ToDataFrame)]
struct Fill {
    #[df_derive(as_str)]
    venue: Venue,                      // AsRef<str>    -> String column
    #[df_derive(as_string)]
    status: Status,                    // Display       -> String column
    #[df_derive(as_binary)]
    payload: Vec<u8>,                  // AsRef<[u8]>   -> Binary column
    #[df_derive(decimal(precision = 18, scale = 6))]
    price: Decimal,                    // Decimal(18, 6)
    #[df_derive(time_unit = "ns")]
    at: DateTime<Utc>,                 // Datetime(Nanoseconds)
    #[df_derive(flatten)]
    key: OrderKey,                     // key's columns, without the `key.` prefix
    #[df_derive(skip)]
    cache: Scratch,                    // no column
}
```

- **`as_str`**: borrow through `AsRef<str>`. No formatting, no temporary.
- **`as_string`**: format with `Display` into one reused scratch buffer.
- **`as_binary`**: borrow through `AsRef<[u8]>`. Without it, `Vec<u8>` is a
  `List(UInt8)`.
- **`decimal(precision = P, scale = S)`**: store any `Decimal128Encode` value
  as `Decimal(P, S)`. Precision is 1 to 38 and scale may not exceed it.
- **`time_unit = "ms" | "us" | "ns"`**: the unit of a datetime or duration
  column.
- **`flatten`** / **`flatten(prefix = "p")`**: splice a nested struct's
  columns into the parent with no prefix, or under an explicit one. The field
  must be a struct or tuple of columns; a plain value has no names of its own
  to keep. Duplicate column names are reported as an error.
- **`skip`**: leave the field out. Its type does not have to be supported.
- **`with = "Builder"`**: use your own `ColumnBuilder` for the value.

The encoding attributes apply to the innermost value. `Option`, `Vec`, `Box`,
`Rc`, `Arc`, `&` and `Cow` written in the field's type are kept around it, so
`#[df_derive(as_str)]` on `Option<Vec<Venue>>` gives a nullable list of
strings. A field has at most one encoding, and tuples cannot carry one: move
the value into a named struct instead.

## Declaring An Encoding Once

A type that should always be stored the same way can say so itself. Every
field of that type then needs no attribute.

**A value type** names its builder with one impl:

```rust
use df_derive::dataframe::{Columnar, builders::{DecimalBuilder, StrBuilder}};

// An enum that is always a string column (needs `AsRef<str>`).
impl Columnar for Currency {
    type Builder = StrBuilder;
}

// A decimal type that is always `Decimal(38, 10)` (needs `Decimal128Encode`).
impl Columnar for Amount {
    type Builder = DecimalBuilder<38, 10>;
}
```

**A single-field wrapper** can be transparent. It produces exactly its field's
column, named after whichever field holds the wrapper:

```rust
#[derive(ToDataFrame)]
#[df_derive(transparent)]
struct Price(Decimal);

#[derive(ToDataFrame)]
struct Candle {
    close: Price,           // one `close` column, not `close.field_0`
    high: Option<Price>,
}
```

A transparent wrapper is a value, not a row, so it does not get
`to_dataframe` of its own.

## Projections

When a type's columns are not its fields, convert it to a row struct and
derive on that. `via` does the conversion one value at a time, with no
intermediate collection, and the row may borrow from the value:

```rust
#[derive(ToDataFrame)]
#[df_derive(via = "InstrumentRow<'_>")]
struct Instrument { /* ... */ }

#[derive(ToDataFrame)]
struct InstrumentRow<'a> {
    symbol: &'a str,
    exchange: Option<&'a str>,
    key: String,            // computed at export time
}

impl<'a> From<&'a Instrument> for InstrumentRow<'a> { /* ... */ }
```

This is also how an enum becomes several columns: give it a row struct with a
kind column and optional columns for the variant data. `try_via` uses
`TryFrom<&T>` instead and turns a failed conversion into an encoding error.

Enums cannot derive `ToDataFrame` directly. Store one as a single string
column with `as_str` / `as_string` or an `impl Columnar`, or spread it over
columns with `via`.

## Errors

A value that cannot be stored produces a `PolarsError` naming the row and the
field path, for example:

```text
Event[1].stamp.at: df-derive: DateTime<Tz> value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)
```

Fallible encodings are nanosecond timestamps, durations that overflow `i64`,
decimals that do not fit the declared precision and scale, `as_string` values
whose `Display` fails, and `try_via` conversions.

## Column Naming

- A field uses its Rust name; raw identifiers drop the `r#`.
- Nested structs use dot notation recursively, such as `address.city`.
- Tuple structs and tuple fields use `field_0`, `field_1`, and so on.
- `flatten` drops the field's own name; `flatten(prefix = "home")` replaces it.
- Wrappers do not add a segment: `quotes: Vec<Quote>` gives `quotes.close`.
- Maps add `key` and `value`.

## Limitations

- Sets and `VecDeque` / `LinkedList` are not supported; use `Vec<T>`.
- A row type cannot contain itself, directly or through wrappers: a
  `DataFrame` has a fixed set of columns.
- A type's columns are fixed at compile time. Data whose column set is only
  known at run time needs a map field or a hand-built frame.
- Nested `Option` layers collapse to one null, so `None` and `Some(None)` are
  indistinguishable in the output.
- `HashMap` pairs are listed in the map's own iteration order.
- Mutable references are not supported as fields.
- A struct that has both lifetime and type parameters cannot be the row type
  of a `via` projection.

## Crate Layout

- **`df-derive`**: the crate to depend on. It re-exports the derive macro and
  the runtime as `df_derive::dataframe`.
- **`df-derive-core`**: the runtime: the `Columnar`, `ColumnBuilder`,
  `ToDataFrame` and `Decimal128Encode` traits and the builders.
- **`df-derive-macros`**: the derive macro.

Generated code refers to the runtime as `::df_derive::dataframe`. A crate that
re-exports it under another path (or a renamed dependency) says where it is:

```rust
#[derive(ToDataFrame)]
#[df_derive(crate = "::my_models")]   // uses ::my_models::dataframe
struct Row { /* ... */ }
```

Types derived in different crates share one trait identity through
`df-derive-core`, so they nest in each other freely.

## Custom Decimal Types

`Decimal128Encode` has one method, which returns a value's `i128` mantissa at
a target scale. Implement it for your decimal type, rounding half-to-even when
scaling down so the stored value matches Polars' own decimal parsing, and
return `None` when the value does not fit. Then either annotate fields with
`decimal(precision = P, scale = S)` or declare the encoding once with
`impl Columnar` as shown above.

## Custom Builders

`ColumnBuilder<T>` is the whole contract between a type and its columns. It
has five required methods: describe the schema, create a builder, push a
value, push a placeholder for a null, and hand the finished columns out.
Implement it to store a type in a way the built-in encodings do not cover,
then select it with `with = "..."` or make it the type's default with
`impl Columnar`. A builder can wrap one of the provided builders and delegate
to it.

The remaining methods have defaults and exist for speed: `extend_from`
appends a block of values at once, and `measure` with `with_capacities` lets
list buffers be allocated at their final size.

## Features

- **`rust_decimal`** (default): `rust_decimal::Decimal` support.
- **`chrono`** (default): `chrono` date and time support.

## Compatibility

- **Rust edition**: 2024
- **Minimum supported Rust version**: 1.95, set by the Polars 0.55 dependency
  graph.
- **Polars**: 0.55

## Benchmarks

Criterion benches in `df-derive/benches/` cover flat, wide, nested, optional,
list, string, decimal, datetime, tuple, generic and smart-pointer shapes.
A head-to-head comparison with hand-written Polars, `serde_arrow` and
`polars-row-derive` lives in
[comparison-bench/README.md](comparison-bench/README.md); its report was
generated against an earlier release.

Performance is continuously monitored with
[Bencher](https://bencher.dev/perf/df-derive).

### How slices are encoded

A slice is encoded in blocks of rows, and within a block each column buffer
is filled by its own loop. Only one buffer is written at a time, and a
block's rows are read from memory once and from cache after that. Filling
many column buffers from a single loop over all rows is much slower for wide
rows, because the buffers compete for the same cache sets.

A block holds 16 KiB of rows, half of the smallest first-level data cache in
common use, so the scheme does not depend on a particular CPU. On the one
machine it was measured on (Apple Silicon), 16 KiB and 128 KiB blocks
performed within about 5% of each other across flat, wide, nested, optional
and string-heavy rows. Blocks of 2 KiB were up to 40% slower on very wide
rows, and 1 MiB blocks, larger than the cache, were 40% to 110% slower on
wide and numeric rows.

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
