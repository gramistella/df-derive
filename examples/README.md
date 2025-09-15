# df-derive Examples

This directory contains runnable examples demonstrating the features of `df-derive`.

## Running Examples

Each example can be run using Cargo's built-in example runner:

```bash
cargo run --example <example_name>
```

## Available Examples

### `quickstart.rs`
Basic usage example showing how to derive `ToDataFrame` on a simple struct and convert both single values and slices to DataFrames.

```bash
cargo run --example quickstart
```

### `nested.rs`
Demonstrates nested structs and how they are flattened with dot notation in the resulting DataFrame columns.

```bash
cargo run --example nested
```

### `vec_custom.rs`
Shows how `Vec<T>` fields become Polars `List` columns, with nested structs creating multiple list columns.

```bash
cargo run --example vec_custom
```

### `tuple.rs`
Example of tuple structs and how their fields are named (`field_0`, `field_1`, etc.).

```bash
cargo run --example tuple
```

### `datetime_decimal.rs`
Demonstrates support for `chrono::DateTime<Utc>` and `rust_decimal::Decimal` types.

```bash
cargo run --example datetime_decimal
```

### `as_string.rs`
Shows the `#[df_derive(as_string)]` attribute for converting enums and other types to strings.

```bash
cargo run --example as_string
```

## What Each Example Shows

- **DataFrame output**: The actual Polars DataFrame structure
- **Schema information**: Column names and data types
- **Different use cases**: From simple structs to complex nested data with lists

All examples use the same trait definitions to demonstrate the complete API surface of the generated code.
