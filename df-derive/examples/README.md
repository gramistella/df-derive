# df-derive Examples

This directory contains runnable examples demonstrating the features of `df-derive`.

## Running Examples

Each example can be run using Cargo's built-in example runner:

```bash
cargo run -p df-derive --example <example_name>
```

## Available Examples

### `quickstart.rs`
Basic usage example showing how to derive `ToDataFrame` on a simple struct and convert both single values and slices to DataFrames.

```bash
cargo run -p df-derive --example quickstart
```

### `nested.rs`
Demonstrates nested structs and how they are flattened with dot notation in the resulting DataFrame columns.

```bash
cargo run -p df-derive --example nested
```

### `vec_custom.rs`
Shows how `Vec<T>` fields become Polars `List` columns, with nested structs creating multiple list columns.

```bash
cargo run -p df-derive --example vec_custom
```

### `tuple.rs`
Example of tuple structs and how their fields are named (`field_0`, `field_1`, etc.).

```bash
cargo run -p df-derive --example tuple
```

### `datetime_decimal.rs`
Demonstrates support for `chrono::DateTime<Utc>` and `rust_decimal::Decimal` types.

```bash
cargo run -p df-derive --example datetime_decimal
```

### `as_string.rs`
Shows the `#[df_derive(as_string)]` attribute for converting enums and other types to strings.

```bash
cargo run -p df-derive --example as_string
```

### `generics.rs`
Demonstrates generic struct support added in v0.3.0: type-parametric structs, default type parameters, multiple generics, the unit type `()` as a zero-column payload, and depth-1 wrappers (`Option<T>` / `Vec<T>`) over a generic parameter.

```bash
cargo run -p df-derive --example generics
```

### `nested_options.rs`
Demonstrates `Option<Option<Struct>>` field handling, including how `Some(None)` and `None` collapse to the same null in the output frame.

```bash
cargo run -p df-derive --example nested_options
```

### `deep_vec.rs`
Demonstrates deep `Vec<Vec<Vec<T>>>` nesting and how each `Vec` layer becomes a nested `List` in the output schema.

```bash
cargo run -p df-derive --example deep_vec
```

### `multi_option_vec.rs`
Demonstrates multi-`Option` wrappers above a `Vec` (e.g. `Option<Option<Vec<T>>>`). Consecutive `Option` layers above a `Vec` collapse to a single list-level validity bit, so `Some(None)` and `None` are indistinguishable in the resulting column.

```bash
cargo run -p df-derive --example multi_option_vec
```

### `nested_generics.rs`
Demonstrates a generic struct used as a nested field (both as a flattened scalar nested type and inside a `Vec<...>` list column).

```bash
cargo run -p df-derive --example nested_generics
```

## What Each Example Shows

- **DataFrame output**: The actual Polars DataFrame structure
- **Schema information**: Column names and data types
- **Different use cases**: From simple structs to complex nested data with lists

The examples use the default `df-derive` facade runtime unless they are demonstrating a custom-runtime pattern.
