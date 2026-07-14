# df-derive comparison benchmark

This directory is the reproducible benchmark artifact behind the short Reddit writeup. It is meant to answer a narrow question:

> If the target is an in-memory Polars `DataFrame` with a rich schema, how much code and time does it take to get there from Rust structs?

It compares `df-derive` with three alternatives:

- hand-written Polars column construction
- `serde_arrow` followed by an Arrow to Polars bridge
- `polars-row-derive` followed by post-processing to recover the target schema

The historical published numbers are in [BENCHMARK.md](BENCHMARK.md). They
were captured with df-derive 0.3.1 and Polars 0.53 and are retained as a
measured artifact, not as a current df-derive 0.4 / Polars 0.54 baseline. The
current harness source targets the dependencies in this directory's
`Cargo.toml`.

The code blocks below are excerpts from [src/main.rs](src/main.rs). The complete executable implementation is the source file, not this README.

## Quick start

From this directory:

```sh
cargo run --release -- --smoke
cargo run --release -- --probe
cargo run --release
```

What each command does:

| Command | Purpose |
| --- | --- |
| `cargo run --release -- --smoke` | Checks rich paths against `df-derive` and the flat pair against each other for shape, column order, dtypes, and values. |
| `cargo run --release -- --probe` | Prints the schema that `serde_arrow::from_type` infers for the flattened row. |
| `cargo run --release` | Runs the full benchmark, prints schema/head/timings, and rewrites `BENCHMARK.md`. |

The full benchmark intentionally excludes data generation from timings. It generates each input batch first, then times only the conversion into a Polars `DataFrame`.

## Files

| File | Role |
| --- | --- |
| [Cargo.toml](Cargo.toml) | Isolated dependency set for the comparison. |
| [Cargo.lock](Cargo.lock) | Locked versions for reproducibility. |
| [src/main.rs](src/main.rs) | The full benchmark harness and all conversion implementations. |
| [BENCHMARK.md](BENCHMARK.md) | Historical generated report from the df-derive 0.3.1 / Polars 0.53 release run. |

This crate is intentionally outside the main workspace by declaring its own `[workspace]` in `Cargo.toml`. That keeps dependency and feature choices explicit.

## Target schema

All rich-schema paths must produce the same 17 Polars columns:

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

The source model is deliberately richer than a flat benchmark row:

```rust
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

## Fairness checks

Before timing, the smoke test builds each rich-schema alternative and compares it to the `df-derive` DataFrame:

```rust
fn ensure_same_frame(label: &str, left: &DataFrame, right: &DataFrame) -> Result<()> {
    anyhow::ensure!(left.shape() == right.shape(), "{label} shape mismatch");
    anyhow::ensure!(
        left.get_column_names() == right.get_column_names(),
        "{label} column names differ"
    );

    let left_dtypes = left
        .columns()
        .iter()
        .map(|column| column.dtype().clone())
        .collect::<Vec<_>>();
    let right_dtypes = right
        .columns()
        .iter()
        .map(|column| column.dtype().clone())
        .collect::<Vec<_>>();

    anyhow::ensure!(left_dtypes == right_dtypes, "{label} dtypes differ");
    anyhow::ensure!(left.equals_missing(right), "{label} values differ");
    Ok(())
}
```

The benchmark is not comparing "whatever each crate naturally emits". It is comparing the cost of reaching the same Polars frame. The smoke path also checks the flat `df-derive` and `polars-row-derive` outputs against each other.

## Measured paths

### df-derive

This is the direct derived columnar conversion from the original nested Rust model:

```rust
let native_rows = generate_showcase_rows(rows);

measure(Approach::DfDerive, rows, || {
    <ShowcaseRow as Columnar>::encode(black_box(native_rows.as_slice()))
})?;
```

`Columnar::encode` is the checked public batch entry point. The derive emits a
hidden `ColumnarSpec` with explicit schema composition and column encoding;
the blanket runtime owns the one-shot input iterator and validates its
`ColumnSink` output before constructing the outer frame. Generated encoders
currently stabilize borrowed rows once in a `Vec<&ShowcaseRow>` for their
shape-dependent passes. `ToDataFrame::schema()` uses the explicit schema path,
not an empty encoding.

The user-facing form is the same conversion through the slice extension:

```rust
let df = rows.as_slice().to_dataframe()?;
```

### hand-written Polars

The manual baseline collects each output column explicitly. Decimal, datetime, Binary, and list columns need separate handling:

```rust
for row in items {
    ids.push(row.id);
    symbols.push(row.symbol.as_str());
    prices.push(decimal_mantissa_for_polars(&row.price)?);
    timestamps.push(row.ts.timestamp_micros());
    fills_venue.push(string_list_anyvalue(
        row.fills.iter().map(|fill| fill.venue.as_str()),
    ));
    marks.push(f64_list_anyvalue(row.marks.iter().copied()));
}

let price_series = Int128Chunked::from_vec("price".into(), prices)
    .into_decimal_unchecked(18, 6)
    .into_series();
let timestamp_series = Series::new("ts".into(), &timestamps)
    .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
let fingerprint_series = BinaryChunked::from_iter_values(
    "fingerprint".into(),
    items.iter().map(|row| row.fingerprint.as_slice()),
)
.into_series();

DataFrame::new(items.len(), vec![
    Series::new("id".into(), &ids).into(),
    Series::new("symbol".into(), &symbols).into(),
    price_series.into(),
    timestamp_series.into(),
    fingerprint_series.into(),
    Series::new("fills.venue".into(), &fills_venue).into(),
    Series::new("marks".into(), &marks).into(),
])?;
```

This is the readable baseline. It is not meant to represent the fastest possible hand-written Polars implementation.

### serde_arrow

The `serde_arrow` path uses a flattened serialization model. The original nested `ShowcaseRow` is converted into `SerdeArrowRow`, with semantic choices made before serialization:

```rust
#[derive(Clone, Serialize, Deserialize)]
struct SerdeArrowRow {
    id: u64,
    symbol: String,
    notional: f64,
    quantity: u64,
    live: bool,
    maybe_note: Option<String>,
    price: String,
    #[serde(with = "chrono::serde::ts_microseconds")]
    ts: DateTime<Utc>,
    side: String,
    #[serde(with = "serde_bytes")]
    fingerprint: Vec<u8>,
    #[serde(rename = "risk.sector")]
    risk_sector: String,
    #[serde(rename = "risk.beta")]
    risk_beta: f64,
    #[serde(rename = "risk.hedged")]
    risk_hedged: bool,
    #[serde(rename = "fills.venue")]
    fills_venue: Vec<String>,
    #[serde(rename = "fills.size")]
    fills_size: Vec<u64>,
    #[serde(rename = "fills.rebate_bps")]
    fills_rebate_bps: Vec<f64>,
    marks: Vec<f64>,
}
```

The schema is explicit because `serde_arrow::from_type` traces structure, not the target semantic schema:

```rust
let fields = Vec::<FieldRef>::from_value(&json!([
    {"name": "id", "data_type": "U64"},
    {"name": "symbol", "data_type": "Utf8"},
    {"name": "price", "data_type": "Decimal128(18, 6)"},
    {"name": "ts", "data_type": "Timestamp(Microsecond, None)"},
    {"name": "fingerprint", "data_type": "Binary"},
    {
        "name": "fills.venue",
        "data_type": "List",
        "children": [{"name": "element", "data_type": "Utf8"}],
    },
]))?;
```

Two bridge costs are measured:

```rust
measure(Approach::SerdeArrowIpc, rows, || {
    serde_arrow_to_polars_ipc(black_box(&serde_rows), black_box(&fields))
})?;

measure(Approach::SerdeArrowFfi, rows, || {
    serde_arrow_to_polars_ffi(black_box(&serde_rows), black_box(&fields))
})?;
```

The IPC bridge serializes the Arrow `RecordBatch` to bytes and reparses it through Polars:

```rust
let batch = serde_arrow::to_record_batch(fields, items)?;
let mut bytes = Vec::new();
let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema())?;
writer.write(&batch)?;
writer.finish()?;
let df = IpcStreamReader::new(Cursor::new(bytes)).finish()?;
```

The FFI bridge avoids IPC, but it uses unsafe C Data Interface glue between `arrow-rs` and `polars-arrow`.

### polars-row-derive

`polars-row-derive` is measured in two ways.

The rich-schema path uses a flat derived row, then post-processes back to the target schema:

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
```

```rust
let prices = items.iter().map(|row| row.flat.price).collect::<Vec<_>>();
let flat_rows = items.iter().map(|row| row.flat.clone()).collect::<Vec<_>>();
let mut df = flat_rows.into_iter().to_dataframe()?;

df.with_column(decimal_series("price", prices).into())?;
replace_with_cast(
    &mut df,
    "ts",
    &DataType::Datetime(TimeUnit::Microseconds, None),
)?;
df.with_column(binary_series(
    "fingerprint",
    items.iter().map(|row| row.fingerprint.as_slice()),
))?;
df.with_column(Series::new("fills.venue".into(), &fills_venue).into())?;
df.with_column(Series::new("marks".into(), &marks).into())?;
df.rename("risk_sector", "risk.sector".into())?;
df.select(OUTPUT_COLUMNS)?;
```

The flat scalar path is a cleaner row-vs-columnar check. It excludes nested lists, Binary, datetime casts, decimal casts, and rich-schema renames. It still includes the per-call clone required by the consuming `polars-row-derive` iterator API:

```rust
measure(Approach::FlatDfDerive, rows, || {
    <RowDeriveFlat as Columnar>::encode(black_box(flat_df_rows.as_slice()))
})?;

measure(Approach::FlatPolarsRowDerive, rows, || {
    flat_row_derive_to_dataframe(black_box(flat_row_derive_rows.as_slice()))
})?;
```

## Timing loop

The harness uses a simple fixed-iteration loop. It does not use Criterion for these published numbers because the goal is a direct, current-machine head-to-head report with explicit generated input batches:

```rust
fn measure<F>(approach: Approach, rows: usize, mut f: F) -> Result<BenchStats>
where
    F: FnMut() -> PolarsResult<DataFrame>,
{
    for _ in 0..WARMUPS {
        let df = f().with_context(|| format!("warmup {}", approach.label()))?;
        black_box(df.shape());
    }

    let mut durations = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let df = f().with_context(|| format!("bench {}", approach.label()))?;
        black_box(df.shape());
        durations.push(start.elapsed());
    }
    durations.sort_unstable();

    Ok(BenchStats {
        approach,
        rows,
        median: durations[durations.len() / 2],
        min: durations[0],
    })
}
```

Current constants:

```rust
const ROW_COUNTS: [usize; 3] = [1_000, 100_000, 1_000_000];
const WARMUPS: usize = 3;
const ITERS: usize = 21;
```

## Interpreting the result

The safe claim is:

> For this rich schema, `df-derive` was the fastest measured path to an in-memory Polars `DataFrame`, and it required the least schema plumbing.

What this does not claim:

- It does not prove that generated code beats the fastest possible hand-written Polars. The manual baseline is intentionally readable, especially around list columns.
- It does not claim `df-derive` is better than `serde_arrow` when the target is Arrow or Parquet. The `serde_arrow` numbers include the cost of ending in Polars.
- It does not use the rich-schema `polars-row-derive` number as a pure row-vs-columnar claim. The flat scalar benchmark exists for that comparison, and its row-derive number still includes the clone forced by the consuming iterator API.

## Updating the benchmark

When revisiting this benchmark:

1. Change the target shape in `ShowcaseRow` and `showcase_row`.
2. Update every alternative until `cargo run --release -- --smoke` passes.
3. Run `cargo run --release`.
4. Commit `src/main.rs`, `Cargo.toml`, `Cargo.lock`, and the regenerated `BENCHMARK.md` together.

If an alternative gets a better implementation, keep the old caveat visible in the commit message or report. The point of this directory is not to protect a benchmark lead, but to make the comparison auditable.
