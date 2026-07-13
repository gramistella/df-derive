# df-derive head-to-head

> **Historical report:** these measurements were captured with df-derive
> 0.3.1 and Polars 0.53. The report is retained as the published measurement
> artifact; it is not a benchmark of the current df-derive 0.4 / Polars 0.54
> dependency set.

## Environment

- Date: Mon Jun  1 20:10:02 UTC 2026
- Host: Darwin Giovannis-MBP.fritz.box 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64
- CPU: Apple M5 Pro
- RAM: 64.0 GiB
- rustc: rustc 1.95.0 (59807616e 2026-04-14)
- cargo: cargo 1.95.0 (f2d3ce0bd 2026-03-21)
- Crates: df-derive 0.3.1, polars 0.53.0, polars-arrow 0.53.0, serde_arrow 0.14.1, arrow 57.3.1, polars-row-derive 0.1.0, rust_decimal 1.42.0
- Harness: release build, 3 warmups, 21 measured iterations, median and min reported, data generation excluded.

## API confirmation

- Public import: `use df_derive::prelude::*;`.
- Derive: `#[derive(ToDataFrame)]` on structs and tuple structs. Nested custom structs must implement `Columnar`, normally by deriving `ToDataFrame`.
- Runtime primitive: derived types implement only `Columnar::encode<B>(&B)` for `B: RowBatch<Self>`; the runtime supplies `RowBatch` for `[T]` and `[&T]`.
- Blanket API: every `Columnar` type receives `ToDataFrame::{to_dataframe, empty_dataframe, schema}` (`schema()` returns `SchemaRef`), plus the `ToDataFrameVec` slice extension.
- Field attributes verified from the current README/docs: `skip`, `flatten`, `flatten(prefix = "...")`, `as_string`, `as_str`, `as_binary`, `decimal(precision = N, scale = S)`, and `time_unit = "ms" | "us" | "ns"`.
- Sources checked: [docs.rs `df-derive`](https://docs.rs/df-derive), [GitHub README](https://github.com/gramistella/df-derive), and the local README in this checkout.

## Showcase shape

The df-derive model is deliberately not a toy row: it has scalar columns, `Option`, decimal, microsecond datetime, enum-as-string, binary bytes, a nested struct, a `Vec<Nested>` split into list columns, and a `Vec<f64>` list column.

```rust
use df_derive::prelude::*;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

#[derive(Clone, ToDataFrame)]
struct RiskBlock {
    sector: String,
    beta: f64,
    hedged: bool,
}

#[derive(Clone, ToDataFrame)]
struct Fill {
    venue: String,
    size: u64,
    rebate_bps: f64,
}

#[derive(Clone, ToDataFrame)]
struct ShowcaseRow {
    id: u64,
    symbol: String,
    notional: f64,
    quantity: u64,
    live: bool,
    maybe_note: Option<String>,
    #[df_derive(decimal(precision = 18, scale = 6))]
    price: Decimal,
    #[df_derive(time_unit = "us")]
    ts: DateTime<Utc>,
    #[df_derive(as_string)]
    side: Side,
    #[df_derive(as_binary)]
    fingerprint: Vec<u8>,
    risk: RiskBlock,
    fills: Vec<Fill>,
    marks: Vec<f64>,
    #[df_derive(skip)]
    cache_key: u64,
}
```

### Schema

```text
id: UInt64
symbol: String
notional: Float64
quantity: UInt64
live: Boolean
maybe_note: String
price: Decimal(18, 6)
ts: Datetime('μs')
side: String
fingerprint: Binary
risk.sector: String
risk.beta: Float64
risk.hedged: Boolean
fills.venue: List(String)
fills.size: List(UInt64)
fills.rebate_bps: List(Float64)
marks: List(Float64)
```

### `df.head()`

```text
shape: (5, 17)
┌─────┬────────┬──────────┬──────────┬───┬────────────────┬────────────┬───────────────┬───────────┐
│ id  ┆ symbol ┆ notional ┆ quantity ┆ … ┆ fills.venue    ┆ fills.size ┆ fills.rebate_ ┆ marks     │
│ --- ┆ ---    ┆ ---      ┆ ---      ┆   ┆ ---            ┆ ---        ┆ bps           ┆ ---       │
│ u64 ┆ str    ┆ f64      ┆ u64      ┆   ┆ list[str]      ┆ list[u64]  ┆ ---           ┆ list[f64] │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆ list[f64]     ┆           │
╞═════╪════════╪══════════╪══════════╪═══╪════════════════╪════════════╪═══════════════╪═══════════╡
│ 0   ┆ SYM000 ┆ 10000.0  ┆ 100      ┆ … ┆ ["VENUE0"]     ┆ [50]       ┆ [-0.2]        ┆ [99.5,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.0,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.5]    │
│ 1   ┆ SYM001 ┆ 10001.25 ┆ 101      ┆ … ┆ ["VENUE1",     ┆ [51, 52]   ┆ [-0.15, -0.1] ┆ [99.6,    │
│     ┆        ┆          ┆          ┆   ┆ "VENUE2"]      ┆            ┆               ┆ 100.1,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.6]    │
│ 2   ┆ SYM002 ┆ 10002.5  ┆ 102      ┆ … ┆ ["VENUE2"]     ┆ [52]       ┆ [-0.1]        ┆ [99.7,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.2,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.7]    │
│ 3   ┆ SYM003 ┆ 10003.75 ┆ 103      ┆ … ┆ ["VENUE3",     ┆ [53, 54]   ┆ [-0.05, 0.0]  ┆ [99.8,    │
│     ┆        ┆          ┆          ┆   ┆ "VENUE0"]      ┆            ┆               ┆ 100.3,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.8]    │
│ 4   ┆ SYM004 ┆ 10005.0  ┆ 104      ┆ … ┆ ["VENUE0"]     ┆ [54]       ┆ [0.0]         ┆ [99.9,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.4,    │
│     ┆        ┆          ┆          ┆   ┆                ┆            ┆               ┆ 100.9]    │
└─────┴────────┴──────────┴──────────┴───┴────────────────┴────────────┴───────────────┴───────────┘
```

## Equivalent implementations

### df-derive

The whole batch conversion is the derived columnar path:

```rust
let df = rows.as_slice().to_dataframe()?;
// or, explicitly:
let df = <ShowcaseRow as Columnar>::encode(rows.as_slice())?;
```

### hand-written Polars

The baseline manually collects every output column. Decimal and datetime require explicit mantissa/timestamp conversion and casts; binary needs a binary chunked array; the list columns are assembled as per-row inner `Series` values.

```rust
let prices: Vec<i128> = rows
    .iter()
    .map(|row| Decimal128Encode::try_to_i128_mantissa(&row.price, 6).unwrap())
    .collect();
let price = Int128Chunked::from_vec("price".into(), prices)
    .into_decimal_unchecked(18, 6)
    .into_series();

let ts_micros: Vec<i64> = rows.iter().map(|row| row.ts.timestamp_micros()).collect();
let ts = Series::new("ts".into(), &ts_micros)
    .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;

let fingerprint = BinaryChunked::from_iter_values(
    "fingerprint".into(),
    rows.iter().map(|row| row.fingerprint.as_slice()),
)
.into_series();

let df = DataFrame::new(rows.len(), vec![
    Series::new("id".into(), &ids).into(),
    Series::new("symbol".into(), &symbols).into(),
    price.into(),
    ts.into(),
    fingerprint.into(),
    Series::new("risk.sector".into(), &risk_sector).into(),
    Series::new("fills.venue".into(), &fills_venue_lists).into(),
    Series::new("marks".into(), &mark_lists).into(),
])?;
```

### serde_arrow

The working serde_arrow version is a flattened serialization model, not the original nested domain struct. The enum is stringified before serialization, decimal is serialized as a string for Arrow `Decimal128`, `DateTime<Utc>` uses chrono's microsecond serializer, bytes use `serde_bytes`, and the final Arrow schema is explicit because the `from_type` probe traces a structural schema, not the semantic Polars schema. The benchmark reports both the original IPC bridge and a non-IPC Arrow C Data FFI bridge.

`Vec::<FieldRef>::from_type::<SerdeArrowRow>(...)` result for the full row: it succeeds structurally, but the traced schema is not the target schema. Traced summary: `id: UInt64, symbol: Utf8, notional: Float64, quantity: UInt64, live: Boolean, maybe_note: Utf8, price: Utf8, ts: Int64, side: Utf8, fingerprint: Binary, risk.sector: Utf8, risk.beta: Float64, risk.hedged: Boolean, fills.venue: List(Field { name: "element", data_type: Utf8 }), fills.size: List(Field { name: "element", data_type: UInt64 }), fills.rebate_bps: List(Field { name: "element", data_type: Float64 }), marks: List(Field { name: "element", data_type: Float64 })`.

```rust
#[derive(Serialize, Deserialize)]
struct SerdeArrowRow {
    id: u64,
    symbol: String,
    maybe_note: Option<String>,
    price: String,
    #[serde(with = "chrono::serde::ts_microseconds")]
    ts: DateTime<Utc>,
    side: String,
    #[serde(with = "serde_bytes")]
    fingerprint: Vec<u8>,
    #[serde(rename = "risk.sector")]
    risk_sector: String,
    #[serde(rename = "fills.venue")]
    fills_venue: Vec<String>,
}

let traced_fields = Vec::<FieldRef>::from_type::<SerdeArrowRow>(
    TracingOptions::default()
        .sequence_as_large_list(false)
        .strings_as_large_utf8(false)
        .bytes_as_large_binary(false),
)?;
// For this full row, from_type does not produce the semantic schema we need.
drop(traced_fields);
let fields = Vec::<FieldRef>::from_value(&json!([
    {"name": "id", "data_type": "U64"},
    {"name": "symbol", "data_type": "Utf8"},
    {"name": "price", "data_type": "Decimal128(18, 6)"},
    {"name": "ts", "data_type": "Timestamp(Microsecond, None)"},
    {"name": "fingerprint", "data_type": "Binary"},
    {"name": "fills.venue", "data_type": "List",
     "children": [{"name": "element", "data_type": "Utf8"}]},
]))?;
let batch = serde_arrow::to_record_batch(&fields, &rows)?;
let df_ipc = arrow_ipc_stream_round_trip_into_polars(batch.clone())?;
let df_ffi = arrow_c_data_ffi_into_polars(batch)?;
```

### polars-row-derive

`polars-row-derive` compiled against Polars 0.53 in this isolated crate. It is row-oriented and has no df-derive-style field attributes, so this benchmark uses a flattened row, then post-processes names and rich dtypes back to the df-derive shape. The iterator API consumes rows; the repeated benchmark therefore clones the rows before each conversion.

```rust
#[derive(Clone, IterToDataFrame)]
struct RowDeriveFlat {
    id: u64,
    symbol: String,
    maybe_note: Option<String>,
    price: i128,
    ts: i64,
    side: String,
    risk_sector: String,
}

let prices = rows.iter().map(|row| row.flat.price).collect::<Vec<_>>();
let flat_rows = rows.iter().map(|row| row.flat.clone()).collect::<Vec<_>>();
let mut df = flat_rows.into_iter().to_dataframe()?;
df.with_column(decimal_series("price", prices).into())?;
df.with_column(
    df.column("ts")?
        .as_materialized_series()
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?,
)?;
df.with_column(binary_series("fingerprint", rows.iter().map(|r| r.fingerprint.as_slice())))?;
df.with_column(Series::new("fills.venue".into(), &fills_venue_lists).into())?;
df.with_column(Series::new("marks".into(), &mark_lists).into())?;
df.rename("risk_sector", "risk.sector".into())?;
// polars-row-derive itself did not build the list columns or Binary column.
```

### flat row speed check

To avoid mixing `polars-row-derive` feature gaps with row-vs-columnar speed, the harness also benchmarks a deliberately flat scalar row through both derives. This flat check excludes nested lists, decimal/datetime/binary post-processing, and rich-schema column renames. The `polars-row-derive` flat path still includes the per-call row clone forced by its consuming iterator API.

```rust
#[derive(Clone, ToDataFrame, IterToDataFrame)]
struct RowDeriveFlat {
    id: u64,
    symbol: String,
    notional: f64,
    quantity: u64,
    live: bool,
    maybe_note: Option<String>,
    price: i128,
    ts: i64,
    side: String,
    risk_sector: String,
    risk_beta: f64,
    risk_hedged: bool,
}

let df_derive_df = <RowDeriveFlat as Columnar>::encode(flat_rows.as_slice())?;
let row_derive_df = flat_rows.into_iter().to_dataframe()?;
```

## Timings

| Approach | Rows | Median | Min |
| --- | ---: | ---: | ---: |
| df-derive | 1000 | 93.209us | 91.250us |
| hand-written Polars | 1000 | 916.208us | 841.042us |
| serde_arrow + IPC bridge | 1000 | 319.958us | 306.833us |
| serde_arrow + C Data FFI bridge | 1000 | 287.917us | 267.625us |
| polars-row-derive + postprocess | 1000 | 775.541us | 746.250us |
| df-derive flat row | 1000 | 22.167us | 22.084us |
| polars-row-derive flat row | 1000 | 93.708us | 90.958us |
| df-derive | 100000 | 7.501ms | 6.764ms |
| hand-written Polars | 100000 | 71.585ms | 70.175ms |
| serde_arrow + IPC bridge | 100000 | 28.483ms | 27.794ms |
| serde_arrow + C Data FFI bridge | 100000 | 20.334ms | 20.033ms |
| polars-row-derive + postprocess | 100000 | 78.189ms | 75.470ms |
| df-derive flat row | 100000 | 2.017ms | 1.917ms |
| polars-row-derive flat row | 100000 | 8.440ms | 8.263ms |
| df-derive | 1000000 | 73.347ms | 72.184ms |
| hand-written Polars | 1000000 | 706.177ms | 696.769ms |
| serde_arrow + IPC bridge | 1000000 | 283.180ms | 280.347ms |
| serde_arrow + C Data FFI bridge | 1000000 | 212.492ms | 210.705ms |
| polars-row-derive + postprocess | 1000000 | 795.398ms | 789.995ms |
| df-derive flat row | 1000000 | 20.837ms | 20.501ms |
| polars-row-derive flat row | 1000000 | 89.877ms | 88.169ms |

Read the rich-schema table with scope: the hand-written Polars baseline is the straightforward boilerplate df-derive saves, not a proof that an expert cannot hand-write lower-level list builders; the `polars-row-derive + postprocess` number includes the work required to reach the same rich schema. The flat-row rows isolate a simpler schema, but the `polars-row-derive` flat path still includes the clone required by its consuming iterator API.

## Findings

### df-derive

- Ergonomics: best fit for a Rust domain struct that should become a Polars DataFrame. The original nested model is the schema; attributes express the exceptional cases locally.
- Performance: measures the generated Polars-native columnar path directly from `&[ShowcaseRow]`. The safe claim is not that no human can hand-write equivalent builders, but that the derive reaches this rich schema without paying a throughput penalty for the abstraction.

### hand-written Polars

- Ergonomics: full control, but every rich dtype forces hand-written conversion code. The code is easy to get subtly wrong because field order, column names, nullability, list inner dtypes, decimal scale, and datetime units all live outside the struct definition.
- Performance: this is the naive, readable hand-written baseline. It builds list columns with one inner `Series` per row, so do not read the multiple as `df-derive beats optimal hand-written Polars by X`; an expert can close that gap by hand-writing lower-level list builders.

### serde_arrow

- Ergonomics: strong if Arrow is the target, but not Polars-native. Matching df-derive's output requires a flattened serialization row, explicit schema fields, serde helper attributes, and a bridge into Polars.
- Type gaps for this shape: enum-as-string is manual, `Vec<u8>` needs `serde_bytes` plus an explicit Binary field, decimal precision/scale is not carried by `rust_decimal::Decimal` and is parsed from a string into Arrow `Decimal128`, and the `Vec<Nested>` shape must be manually split into one list field per nested member.
- Performance: the IPC number includes RecordBatch creation plus serialize/reparse bridge cost. The C Data FFI number is the non-IPC check; it still measures the same end state, an in-memory Polars DataFrame.

### polars-row-derive

- Ergonomics: stale-looking but usable for this probe with Polars 0.53 because the macro expands to `polars::df!`. It does not understand nested flattening, decimal/time/binary attributes, or borrowed batch conversion.
- Type gaps for this shape: column names with dots require post-rename, decimal/datetime require post-casts, and Binary requires rebuilding/replacing the column. Without those post-steps the output is not the same schema.
- Performance: the rich-schema number includes the row-derive conversion, row cloning needed by the consuming iterator API, and post-processing required to reach the same DataFrame schema. The separate flat-row number removes rich-schema post-processing, but still includes the clone required by the consuming iterator API.

## Gotchas encountered

- `serde_arrow::to_record_batch` takes `&T: Serialize`, so passing a bare slice failed because `[SerdeArrowRow]` is unsized. The harness passes `&Vec<SerdeArrowRow>` instead.
- `serde_arrow` could trace the flattened row structurally, but that trace was not the semantic target schema: `price` traced as `Utf8` and `ts` as `Int64`, so Decimal128 precision/scale and Timestamp unit had to be supplied explicitly.
- The non-IPC serde_arrow bridge is possible through Arrow C Data FFI, but it is not a simple `RecordBatch -> DataFrame` public API. The harness uses a small unsafe adapter between upstream arrow-rs FFI structs and polars-arrow FFI structs, then verifies dtype parity before timing it.
- Serializing an `i128` decimal carrier into Arrow `Decimal128(18, 6)` failed; decimal strings worked, but parsing those strings is part of the measured serde_arrow cost.
- `serde_bytes` was needed to keep `Vec<u8>` on the binary path. Without being explicit about bytes, it is easy to accidentally compare a list-of-u8 shape instead of a Binary column.
- `polars-row-derive` compiled with Polars 0.53, but its generated `polars::df!` path could not build the `Vec<Vec<T>>` list columns used by the flattened `Vec<Nested>` representation. Those columns had to be added manually after the derived conversion.
- `polars-row-derive` consumes an iterator of owned rows, so repeated benchmark iterations either consume the dataset or require cloning/collecting a flat row buffer. The benchmark includes that cloning because it is required by the usable API shape here.
- Hand-written Polars is easy to make unfair accidentally. The smoke test checks shape, column order, dtypes, and values against df-derive so the manual, serde_arrow, and row-derive paths all end at the same DataFrame before timing claims are made. It also checks the flat `df-derive` and `polars-row-derive` outputs against each other.
- The hand-written baseline is intentionally the maintainable/obvious version, not the theoretical ceiling. A lower-level hand implementation using list builders should be faster than this baseline and could approach df-derive's generated code.

## Headline takeaway

df-derive gives one derive from a real Rust struct to a correctly typed Polars DataFrame, and in this benchmark it is the fastest measured path to that in-memory Polars result; serde_arrow is credible but needs an Arrow-first adapter layer, and polars-row-derive needs enough post-processing that it stops being a direct alternative for rich schemas.
