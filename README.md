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
df-derive = "0.3"
polars = { version = "0.53", features = ["timezones", "dtype-date", "dtype-time", "dtype-duration", "dtype-decimal"] }
polars-arrow = "0.53"

# If your models use these types:
chrono = { version = "0.4", features = ["serde"] }
rust_decimal = { version = "1.41", features = ["serde"] }
```

`polars-arrow` is still a direct dependency for crates that derive
`ToDataFrame`; generated code uses public Arrow array builders that Polars
does not re-export through `polars::prelude`.

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
`Decimal128Encode`; it also exports the trait as `ToDataFrameTrait` for code
that wants an unambiguous type-namespace alias.

## Crate Layout

This repository uses a serde-like three-crate architecture:

- `df-derive`: the normal facade crate. It re-exports the derive macro from
  `df-derive-macros` and the runtime API from `df-derive-core`.
- `df-derive-core`: a normal library crate that owns the shared
  `dataframe::{ToDataFrame, Columnar, ToDataFrameVec, Decimal128Encode}` trait
  identity, the `()` impls, and the optional reference
  `Decimal128Encode for rust_decimal::Decimal` impl.
- `df-derive-macros`: the proc-macro implementation. Power users can depend
  on this directly and target `df-derive-core`, `paft`, or a custom runtime.

Because `df-derive-core` owns the default trait identity, models derived in
different crates can compose as nested `ToDataFrame` types when they use the
facade/default runtime.

## Generated API

For each struct or tuple struct `T`, the macro generates:

- `impl ToDataFrame for T`
  - `fn to_dataframe(&self) -> PolarsResult<DataFrame>`
  - `fn empty_dataframe() -> PolarsResult<DataFrame>`
  - `fn schema() -> PolarsResult<Vec<(String, DataType)>>`
- `impl Columnar for T`
  - `fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>`
  - `fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>`

The direct `&[Self]` method is generated so top-level slice conversion does
not allocate a temporary `Vec<&Self>`. The borrowed `&[&Self]` method remains
for nested and generic composition.

## Supported Types And Shapes

Container and wrapper support:

- **Named structs**: each field becomes one or more columns.
- **Nested structs**: fields flatten recursively with dot notation.
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
  necessary `ToDataFrame + Columnar` bounds, plus `Decimal128Encode` for
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

Useful field attributes:

- `#[df_derive(skip)]`: omit a field from generated schema and DataFrame output.
- `#[df_derive(as_string)]`: format values with `Display` into a string column.
- `#[df_derive(as_str)]`: borrow via `AsRef<str>` without per-row string allocation.
- `#[df_derive(as_binary)]`: encode byte-buffer shapes as Binary.
- `#[df_derive(decimal(precision = N, scale = S))]`: choose a decimal dtype or opt a custom decimal backend into `Decimal128Encode`.
- `#[df_derive(time_unit = "ms" | "us" | "ns")]`: choose datetime or duration units.

`skip` is useful for caches, source metadata, handles, or unsupported helper
fields that should remain on the Rust struct but not become DataFrame columns.
It is mutually exclusive with conversion attributes because skipped fields are
not analyzed or emitted. Tuple struct fields can be skipped too; remaining
tuple columns keep their original `field_{index}` names.

`as_string` is useful for enums or validated newtypes that should appear as
string columns. If a field already implements `AsRef<str>`, prefer `as_str`:
it borrows through the same columnar buffer used for bare `String`/`&str`
fields and avoids per-row string allocation. The two attributes are mutually
exclusive.

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
struct when you need an attributed field. Nested tuples inside an outer
`Option` or `Vec` are rejected for now; use a named struct for those shapes.

## Column Naming

- Named struct fields use the Rust field name, such as `symbol`.
- Nested structs use dot notation recursively, such as `address.city`.
- `Vec<Nested>` fields use the outer field plus nested field name, such as
  `quotes.close`.
- Tuple-typed fields use `field.field_0`, `field.field_1`, and recurse for
  unwrapped nested tuples.
- Tuple structs use `field_0`, `field_1`, and so on.

## Limitations And Guidance

- Maps and sets such as `HashMap<_, _>` are not supported; use `Vec<(K, V)>`
  or a named row struct when you need a tabular representation.
- All nested custom structs must also derive `ToDataFrame`.
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

If only `trait = "x::ToDataFrame"` is provided, the macro infers
`x::Columnar` and `x::Decimal128Encode` unless those paths are explicitly
overridden.

Without overrides, the macro discovers a `dataframe` module in this order:

1. `paft::dataframe`
2. `paft_utils::dataframe`
3. `df_derive::dataframe`
4. `df_derive_core::dataframe`
5. `crate::core::dataframe`

Discovery uses `proc_macro_crate::crate_name`, so dependency renames are
respected. For example, a dependency declared as
`dfd = { package = "df-derive", version = "0.3" }` is emitted as
`::dfd::dataframe`.

The final `crate::core::dataframe` fallback is for legacy/local runtimes in
crates that use `df-derive-macros` directly without `paft`, `df-derive`, or
`df-derive-core`.

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
df-derive-core = "0.3"
df-derive-macros = "0.3"
polars = "0.53"
polars-arrow = "0.53"
```

```rust
use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
use df_derive_macros::ToDataFrame;

#[derive(ToDataFrame)]
struct Row {
    id: u32,
}
```

Use a custom runtime by providing compatible traits and overriding paths. The
minimum trait surface is:

```rust
mod runtime {
    pub mod dataframe {
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

        pub trait Decimal128Encode {
            fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
        }
    }
}
```

## Decimal Backends

`df-derive-core` provides `Decimal128Encode for rust_decimal::Decimal` behind
the `rust_decimal` feature, which is enabled by default on both `df-derive`
and `df-derive-core`.

To disable it:

```toml
df-derive = { version = "0.3", default-features = false }
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
- **Polars**: 0.53
- **polars-arrow**: 0.53, required as a direct dependency because generated
  code uses public Arrow array builders that Polars does not re-export
- **Polars feature flags**: enable `timezones` for timezone-aware
  `DateTime<Tz>`, `dtype-date` for `NaiveDate`, `dtype-time` for
  `NaiveTime`, `dtype-duration` for duration columns, `dtype-i8` /
  `dtype-i16` / `dtype-u8` / `dtype-u16` for exact small-integer columns,
  `dtype-i128` / `dtype-u128` for 128-bit integer columns, and
  `dtype-decimal` for decimal columns.

## Performance Notes

Using `df_derive::dataframe::Columnar` instead of `paft::dataframe::Columnar`
has no inherent runtime performance penalty. The macro generates the hot
column-building code at the impl site either way; the runtime path only
selects which trait receives the impl.

The generated `columnar_to_dataframe(&[Self])` path avoids the old top-level
`Vec<&Self>` allocation. Nested and generic emitters still use
`columnar_from_refs(&[&Self])` so borrowed composition remains clone-free.

Criterion benches in `df-derive/benches/` cover wide rows, nested structs,
deep Vec shapes, decimals, strings, borrowed data, and tuple fields.

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
