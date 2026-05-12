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

## Supported Shapes

The derive supports named structs, tuple structs, nested structs flattened with
dot notation, `Option<T>`, `Vec<T>` in arbitrary wrapper stacks, tuple-typed
fields, `Box<T>`, `Rc<T>`, `Arc<T>`, borrowed references, and `Cow`.

Common leaf types include strings, bools, signed and unsigned integers
including `i128`/`u128`, `NonZero*` integer types, `f32`, `f64`,
`chrono::DateTime<Tz>`, `chrono::NaiveDateTime`, `chrono::NaiveDate`,
`chrono::NaiveTime`, `std::time::Duration`, `core::time::Duration`,
`chrono::Duration`, and `rust_decimal::Decimal`.

Useful field attributes:

- `#[df_derive(as_string)]`: format values with `Display` into a string column.
- `#[df_derive(as_str)]`: borrow via `AsRef<str>` without per-row string allocation.
- `#[df_derive(as_binary)]`: encode `Vec<u8>`, `&[u8]`, or `Cow<'_, [u8]>` as Binary.
- `#[df_derive(decimal(precision = N, scale = N))]`: choose a decimal dtype or opt a custom decimal backend into `Decimal128Encode`.
- `#[df_derive(time_unit = "ms" | "us" | "ns")]`: choose datetime or duration units.

Enums are not supported as derive targets; use `as_string` or `as_str` on enum
fields. Direct fields of type `()` are rejected, but `()` is supported as a
generic payload and contributes zero columns.

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
`#[df_derive(decimal(precision = N, scale = N))]` on fields that should be
encoded as Polars decimal columns. Implementations must return an `i128`
mantissa rescaled to the requested scale, using round-half-to-even when
scaling down. Returning `None` surfaces as a Polars compute error.

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

## Examples

Run examples with:

```sh
cargo run -p df-derive --example quickstart
cargo run -p df-derive --example nested
cargo run -p df-derive --example datetime_decimal
```

## License

MIT. See `LICENSE`.
