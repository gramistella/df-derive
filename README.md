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
The runtime enables the Polars features in the table below.

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

The derive does not analyse the type of a field that has no attribute. Each
supported type implements the `Columnar` trait, which names a *builder* that
owns the buffers for that type's columns. A derived struct gets a builder
with one member per field, and wrappers compose by wrapping the builder of
what they contain:

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
- **Dates and times** (`chrono` feature, on by default): `DateTime<Tz>` as
  `Datetime(Microseconds, UTC)`, `NaiveDateTime` as `Datetime(Microseconds)`
  with no time zone, `NaiveDate` as `Date`, `NaiveTime` as `Time`,
  `chrono::Duration` / `TimeDelta` as `Duration(Nanoseconds)`.
  `std::time::Duration` is `Duration(Nanoseconds)` with or without the
  feature. Choose another unit for a field with `time_unit`.
- **Decimals** (`rust_decimal` feature, on by default):
  `rust_decimal::Decimal` as `Decimal(38, 10)`. A column has one scale, so a
  value with more than 10 digits after the point is rounded to 10, half to
  even, and a value with more than 28 digits before it is an error. Choose
  another precision and scale for a field with `decimal(...)`.
- **Other derived structs**: flattened into the parent with dot notation.
- **Tuples**: `(A, B)` becomes `field.field_0`, `field.field_1`.
- **`Option<T>`**: the same columns as `T`, nullable.
- **`Vec<T>`, `&[T]`, `Box<[T]>`, `[T; N]`**: the same columns as `T`, each
  wrapped in a `List`. A list of structs becomes one list column per struct
  field. A fixed-size array is a list like the others: its length is not
  part of the column's type.
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
| `DateTime<Tz>` | `Datetime`, time zone `UTC` | `dtype-datetime`, `timezones` |
| `NaiveDateTime` | `Datetime`, no time zone | `dtype-datetime` |
| `NaiveDate` | `Date` | `dtype-date` |
| `NaiveTime` | `Time` | `dtype-time` |
| `std::time::Duration`, `chrono::Duration` | `Duration` | `dtype-duration` |
| `rust_decimal::Decimal`, `decimal(...)` | `Decimal` | `dtype-decimal` |
| `as_binary` | `Binary` | none |

`isize` and `usize` are stored as 64-bit columns on every platform, so a
schema does not depend on the build target. `NonZero*` types use the dtype of
their base integer.

A `DateTime<Tz>` is an instant, so its column is marked as UTC, whatever zone
or offset the value was built with: the instant is kept, the label is not. A
`NaiveDateTime` names no zone, and neither does its column. Polars treats the
two as different types. Code that needs a zone-less column from instants casts
it, `column.cast(&DataType::Datetime(unit, None))`, which leaves the stored
values as they are.

A datetime is stored in microseconds unless its field names another unit.
Microseconds hold every date `chrono` has, so the default cannot fail, and
digits finer than a microsecond are dropped. `time_unit = "ns"` keeps them,
but nanoseconds reach only from the year 1677 to 2262, and a datetime outside
that range is an error. A duration is stored in nanoseconds.

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
    at: DateTime<Utc>,                 // Datetime(Nanoseconds, UTC)
    #[df_derive(flatten)]
    key: OrderKey,                     // key's columns, without the `key.` prefix
    #[df_derive(rename = "qty")]
    quantity: u32,                     // a column named `qty`
    #[df_derive(skip)]
    cache: Scratch,                    // no column
}
```

- **`as_str`**: borrow through `AsRef<str>`. No formatting, no temporary.
- **`as_string`**: format with `Display` into one reused scratch buffer.
- **`as_binary`**: borrow through `AsRef<[u8]>`. Without it, `Vec<u8>` and
  `[u8; N]` are a `List(UInt8)`.
- **`decimal(precision = P, scale = S)`**: store any `Decimal128Encode` value
  as `Decimal(P, S)`. Precision is 1 to 38 and scale may not exceed it. A
  value with more than `S` digits after the point is rounded half to even; a
  value that then needs more than `P` digits is an error.
- **`time_unit = "ms" | "us" | "ns"`**: the unit of a datetime or duration
  column. Digits finer than the unit are dropped.
- **`flatten`** / **`flatten(prefix = "p")`**: splice a nested struct's
  columns into the parent with no prefix, or under an explicit one. The field
  must be a struct or tuple of columns; a plain value has no names of its own
  to keep. Duplicate column names are reported as an error.
- **`rename = "name"`**: use `name` in place of the field's own name. On a
  nested struct it is the prefix of the struct's columns. It combines with an
  encoding attribute, not with `flatten`.
- **`skip`**: leave the field out. Its type does not have to be supported.
- **`with = "Builder"`**: use your own `ColumnBuilder` for the value.

The encoding attributes apply to the innermost value. `Option`, `Vec`,
slices, arrays, `Box`, `Rc`, `Arc`, `&` and `Cow` written in the field's type
are kept around it, so
`#[df_derive(as_str)]` on `Option<Vec<Venue>>` gives a nullable list of
strings. For a `HashMap` or `BTreeMap` the innermost value is the map's
value: `#[df_derive(as_string)]` on `BTreeMap<String, Json>` formats the
values. The keys keep the encoding of their own type, so a key type that is
not supported as it stands needs an `impl Columnar`. A field has at most one
encoding, and tuples cannot carry one: move the value into a named struct
instead.

This is the one place where the derive reads a type instead of asking the
compiler: the wrappers around an encoded value are taken from the type as it
is written. An alias that hides one, such as `type Tags = Vec<Tag>`, is taken
as the value itself.

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

A value that cannot be stored produces a `PolarsError` that says where it is:
the row, every field on the way to the value, and the position inside a list.

```text
Event[1].stamp.at: df-derive: DateTime<Tz> value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)
Order[222].legs[3].price: df-derive: decimal mantissa 123456789 exceeds declared precision 6 for Decimal(6, 2)
```

When several values are bad, the error is about the first bad row, and within
it the first field in declaration order.

Fallible encodings are nanosecond timestamps, durations that overflow `i64`,
decimals that do not fit the declared precision and scale, `as_string` values
whose `Display` fails, and `try_via` conversions.

## Column Naming

- A field uses its Rust name; raw identifiers drop the `r#`.
- Nested structs use dot notation recursively, such as `address.city`.
- Tuple structs and tuple fields use `field_0`, `field_1`, and so on.
- `rename = "qty"` replaces the field's own name.
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
- The zone or offset of a `DateTime<Tz>` is not kept. Its column is UTC, and
  a column has one zone for all of its rows.
- Mutable references are not supported as fields.
- A `via` row that holds a reference to a value of generic type, such as
  `&'_ T`, needs `T: 'static`.

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
return `None` when the value has no `i128` mantissa at that scale. The trait
is arithmetic only: whether the mantissa fits the column's precision is
checked by the column builder. Then either annotate fields with
`decimal(precision = P, scale = S)` or declare the encoding once with
`impl Columnar` as shown above.

## Custom Builders

`ColumnBuilder<T>` is the whole contract between a type and its columns.
Implement it to store a type in a way the built-in encodings do not cover,
then select it with `with = "..."` or make it the type's default with
`impl Columnar`.

A builder is filled one slot at a time: `Some(value)`, or `None` for a
position that an enclosing `Option` records as null. Five methods are
required. `schema` names the columns, `new` creates an empty builder,
`reserve` makes room, `push` appends one slot and `finish` hands the columns
over. A builder usually wraps one of the provided ones and delegates:

```rust
use df_derive::dataframe::builders::PrimitiveBuilder;
use df_derive::dataframe::{Bitmap, ColumnBuilder, Fields};
use polars::prelude::{Column, PlSmallStr, PolarsResult};

/// Stores a price as integer cents.
struct Cents(PrimitiveBuilder<i64>);

impl ColumnBuilder<f64> for Cents {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <PrimitiveBuilder<i64> as ColumnBuilder<i64>>::schema(name, out);
    }

    fn new() -> Self {
        Self(ColumnBuilder::<i64>::new())
    }

    fn reserve(&mut self, additional: usize) {
        ColumnBuilder::<i64>::reserve(&mut self.0, additional);
    }

    fn push(&mut self, value: Option<&f64>) -> PolarsResult<()> {
        let cents = value.map(|value| (value * 100.0).round() as i64);
        self.0.push(cents.as_ref())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        ColumnBuilder::<i64>::finish(self.0, name, validity, out)
    }
}
```

Three more items have defaults that are always correct, and exist to be
overridden for speed. `extend` appends many slots at once, and `measure`
lets a builder that holds lists size their buffer before anything is
appended. `ROW_AT_A_TIME` asks a row's builder to push to this builder one
row at a time, together with the row's other builders that ask the same,
instead of handing it a block of rows as one run. The builders that borrow
a string or bytes set it, because such a value is reached through a pointer
and appended with a call, so a run of its own gains it nothing. A builder
that wraps another one passes the inner builder's answer on.

A builder borrows nothing from the values it reads, so keep their lifetimes
out of its type. The builder for a `Code<'a>` has no lifetime of its own, and
the builder for a generic `Wrapper<T>` is generic over the builder of `T`, as
`OptionBuilder<B>` is, not over `T`. A derived type that holds such a value
keeps one builder for the field and feeds it values of any lifetime.

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
list, string, decimal, datetime, tuple, generic, smart-pointer and projected
shapes. A head-to-head comparison with hand-written Polars, `serde_arrow` and
`polars-row-derive` lives in
[comparison-bench/README.md](comparison-bench/README.md); its report was
generated against an earlier release.

Performance is continuously monitored with
[Bencher](https://bencher.dev/perf/df-derive).

Every figure in this section comes from one machine, an Apple Silicon
laptop. The times belong to that machine. The ratios should carry over to
other hardware only roughly, and have not been checked on x86.

### 0.6 against 0.5.0

The better of two runs on each side:

| Rows | 0.5.0 | 0.6 | Time taken |
| --- | ---: | ---: | ---: |
| 100k rows, 6 numeric columns | 252 µs | 115 µs | 46% |
| 100k rows, 32 numeric columns | 13.51 ms | 1.37 ms | 10% |
| 100k rows, 8 nested structs of 4 columns | 3.91 ms | 1.25 ms | 32% |
| 50k rows, 10 optional columns of mixed types | 648 µs | 515 µs | 79% |
| 50k rows, optional structs two levels deep | 1.11 ms | 696 µs | 63% |
| 100k rows, 5 string columns | 1.52 ms | 1.62 ms | 106% |
| 100k rows, 5 optional string columns | 1.56 ms | 1.51 ms | 97% |
| 100k rows, datetimes and decimals | 615 µs | 452 µs | 73% |
| lists of 6-column structs, 100k structs | 411 µs | 116 µs | 28% |
| 100k rows of a generic struct, numeric payload | 180 µs | 47 µs | 26% |
| 10k rows, lists of 3 to 9 optional decimals | 119 µs | 96 µs | 81% |
| 10k rows, lists of lists of 2 to 5 optional integers | 121 µs | 137 µs | 113% |
| one row of dates, times and durations | 2.26 µs | 1.52 µs | 67% |

Over all 53 benchmarks that both versions can run, 0.6 takes
0.66 times as long as 0.5.0 (geometric mean). 37 are faster by
more than 3%, 5 are within 3%, and 11 are slower, by
13% at most. The slower ones are rows whose lists hold two or three
elements each, and rows that are all strings. Both are explained below.

### How rows are encoded

**Every buffer is allocated once.** Before anything is appended the rows are
measured: builders that hold lists add up their lengths, and each buffer is
then reserved at its final size. The pass costs about 0.2 ns per row.
Without it the buffers behind list columns grow by reallocation, and rows
whose lists hold 16 elements take 18% to 49% longer, rows whose lists
hold 64 take 39% to 68% longer.

**Rows are appended in blocks, one column at a time.** Within a block each
column buffer is filled by its own loop, so only one buffer is being written
at a time and the block's rows are read from memory once and from cache
after that. Feeding every column buffer from a single loop over all rows,
which is what 0.5.0 did, is much slower for wide rows: the first table shows
32 numeric columns at 10% of the time.

A block is 4 KiB of rows and never fewer than 32 rows. 4 KiB is one way of
a 32 KiB eight-way first-level data cache, the smallest in common use, so
the rows of a block never compete with each other for a place in the cache.
The floor keeps the fixed cost of starting each column's loop small next to
the loop. This is how nine of the benchmark targets respond to other block
sizes, as time relative to 4 KiB:

| Rows | 2 KiB | 4 KiB | 8 KiB | 16 KiB | 32 KiB | 128 KiB | 1 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 numeric columns | 1.03 | 1.00 | 1.02 | 1.58 | 1.50 | 1.43 | 2.66 |
| 8 nested structs of 4 columns | 1.00 | 1.00 | 0.99 | 1.55 | 1.47 | 1.50 | 2.71 |
| 6 numeric columns | 0.86 | 1.00 | 1.05 | 1.05 | 1.00 | 1.00 | 2.14 |
| 10 optional columns | 1.02 | 1.00 | 1.04 | 0.98 | 0.99 | 0.97 | 1.48 |
| 5 string columns | 1.01 | 1.00 | 0.97 | 0.96 | 0.95 | 0.95 | 0.95 |
| optional nested structs | 0.99 | 1.00 | 1.00 | 0.98 | 0.98 | 0.98 | 1.07 |
| datetimes and decimals | 0.99 | 1.00 | 1.00 | 0.97 | 0.97 | 1.00 | 1.24 |
| lists of structs | 1.00 | 1.00 | 1.07 | 1.13 | 1.07 | 1.20 | 2.21 |
| all 17 benchmarks in the sweep (geometric mean) | 0.98 | 1.00 | 1.02 | 1.09 | 1.06 | 1.07 | 1.47 |

From 2 to 8 KiB the results are within a few percent of each other. From
16 KiB up, rows with dozens of numeric columns take about half as long
again, because each column's run of writes gets long, and blocks larger
than the cache are slower across the board. Without the 32-row floor,
blocks of 16 rows were erratic: 35% faster than 64 rows on one wide shape
and 30% slower on another.

**String fields are appended a row at a time.** A string is reached through a
pointer and appended with a call, so a loop of its own gains it nothing, and
a row with several strings would be walked once for each of them. The fields
of a row that borrow a string or bytes are fed together instead: one walk
over the rows of the block, before the other fields get their runs. (A value
formatted with `as_string` is not borrowed, and keeps its own run.) For these
fields an `Option` records only the empty slots, in a bitmap that starts out
all set.
This is how 0.5.0 appended every field, and for strings it was right. Fed a
column at a time, rows of five strings took 1.06 to 1.96 times as long as
0.5.0 on five x86 server processors; a row at a time they take 0.99 to 1.08
times as long. A string column that is entirely null is the one case that
pays, by up to 30%.

**Lists are appended one at a time.** The elements of a list sit together in
memory, so each list is handed to the inner builder as one run per buffer,
and a list of one element is simply pushed. Time relative to 0.5.0, which
appended element by element, by the number of elements in each list (frames
of 40,000 elements, each case timed in a process of its own):

| Elements per list | 0 | 1 | 2 | 3 | 4 | 8 | 16 | 64 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| integers | 0.57 | 0.82 | 1.02 | 0.98 | 0.77 | 0.50 | 0.38 | 0.24 | 0.19 |
| optional integers | 0.84 | 1.08 | 1.38 | 1.31 | 0.92 | 0.89 | 0.53 | 0.33 | 0.31 |
| structs of 3 fields | 0.81 | 0.78 | 0.67 | 0.60 | 0.60 | 0.68 | 0.51 | 0.33 | 0.33 |

Lists of two or three scalars are the one shape where the fixed cost of a
run is not repaid.

**A projection never stores the row.** A type encoded through `via` is
converted and pushed one value at a time, so the row lives in registers.
Converting by hand, by collecting the rows into a `Vec` and encoding that,
takes 8% to 17% longer for flat rows and 112% longer when the
projected type sits in a list.

**An error is located afterwards.** The fast path does not track where it
is. When a block fails, its rows are pushed again one at a time into a
scratch builder, and the first one that fails gives the error its row, its
fields and its position in a list.

### Compile time

A library of 50 row types with 12 fields each, and a program that converts
all 50:

| | 0.5.0 | 0.6 |
| --- | ---: | ---: |
| library, debug build | 3.4 s | 3.8 s |
| library, release build | 19.4 s | 8.7 s |
| program, either build | 0.9 s | 1.0 s |
| release binary | 30.9 MB | 30.5 MB |

The encoder for a row type is compiled once, in the crate that defines the
type, so a crate that only converts values pays nothing for it.

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
- **`projection`**: `via` for an enum spread over columns and for a row that
  borrows from the value.
- **`custom_builder`**: a hand-written `ColumnBuilder` selected with `with`.

## License

MIT. See `LICENSE`.
