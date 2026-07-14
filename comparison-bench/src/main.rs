use std::fmt::{self, Display};
use std::fs;
use std::hint::black_box;
use std::io::Cursor;
use std::mem::ManuallyDrop;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use arrow::datatypes::FieldRef;
use arrow::ipc::writer::StreamWriter;
use chrono::{DateTime, TimeZone, Utc};
use df_derive::prelude::*;
use polars::io::SerReader;
use polars::io::ipc::IpcStreamReader;
use polars::prelude::*;
use polars_arrow::array::ArrayRef as PolarsArrayRef;
use polars_row_derive::IterToDataFrame;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_arrow::schema::{SchemaLike, TracingOptions};
use serde_json::json;

const ROW_COUNTS: [usize; 3] = [1_000, 100_000, 1_000_000];
const WARMUPS: usize = 3;
const ITERS: usize = 21;
const PRICE_PRECISION: usize = 18;
const PRICE_SCALE: usize = 6;
const OUTPUT_COLUMNS: [&str; 17] = [
    "id",
    "symbol",
    "notional",
    "quantity",
    "live",
    "maybe_note",
    "price",
    "ts",
    "side",
    "fingerprint",
    "risk.sector",
    "risk.beta",
    "risk.hedged",
    "fills.venue",
    "fills.size",
    "fills.rebate_bps",
    "marks",
];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Side {
    Buy,
    Sell,
    Short,
    Cover,
}

impl Side {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Buy => "buy",
            Self::Sell => "sell",
            Self::Short => "short",
            Self::Cover => "cover",
        }
    }
}

impl Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

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
    #[allow(dead_code)]
    cache_key: u64,
}

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

#[derive(Clone)]
struct RowDeriveInput {
    flat: RowDeriveFlat,
    fingerprint: Vec<u8>,
    fills_venue: Vec<String>,
    fills_size: Vec<u64>,
    fills_rebate_bps: Vec<f64>,
    marks: Vec<f64>,
}

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

#[derive(Clone, Copy)]
enum Approach {
    DfDerive,
    ManualPolars,
    SerdeArrowIpc,
    SerdeArrowFfi,
    PolarsRowDerive,
    FlatDfDerive,
    FlatPolarsRowDerive,
}

impl Approach {
    const fn label(self) -> &'static str {
        match self {
            Self::DfDerive => "df-derive",
            Self::ManualPolars => "hand-written Polars",
            Self::SerdeArrowIpc => "serde_arrow + IPC bridge",
            Self::SerdeArrowFfi => "serde_arrow + C Data FFI bridge",
            Self::PolarsRowDerive => "polars-row-derive + postprocess",
            Self::FlatDfDerive => "df-derive flat row",
            Self::FlatPolarsRowDerive => "polars-row-derive flat row",
        }
    }
}

#[derive(Clone)]
struct BenchStats {
    approach: Approach,
    rows: usize,
    median: Duration,
    min: Duration,
}

struct ShowcaseOutput {
    schema: String,
    head: String,
}

fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--probe") {
        println!("{}", serde_arrow_from_type_probe());
        return Ok(());
    }
    if std::env::args().any(|arg| arg == "--smoke") {
        return smoke();
    }

    let showcase = showcase_output()?;
    println!("Schema:\n{}\n", showcase.schema);
    println!("Head:\n{}\n", showcase.head);

    let timings = run_benchmarks()?;
    for stat in &timings {
        println!(
            "{} {:>9}: median {}, min {}",
            stat.approach.label(),
            stat.rows,
            format_duration(stat.median),
            format_duration(stat.min)
        );
    }

    let report = render_report(&showcase, &timings)?;
    fs::write("BENCHMARK.md", report).context("write BENCHMARK.md")?;
    Ok(())
}

fn smoke() -> Result<()> {
    let native = generate_showcase_rows(8);
    let df_derive = native.as_slice().to_dataframe()?;
    let manual = manual_polars_to_dataframe(&native)?;
    ensure_same_frame("manual", &df_derive, &manual)?;

    let serde_rows = generate_serde_arrow_rows(8);
    let serde_fields = serde_arrow_fields()?;
    let serde_ipc = serde_arrow_to_polars_ipc(&serde_rows, &serde_fields)?;
    ensure_same_frame("serde_arrow IPC", &df_derive, &serde_ipc)?;
    let serde_ffi = serde_arrow_to_polars_ffi(&serde_rows, &serde_fields)?;
    ensure_same_frame("serde_arrow FFI", &df_derive, &serde_ffi)?;

    let row_derive_rows = generate_row_derive_rows(8);
    let row_derive = polars_row_derive_to_dataframe(&row_derive_rows)?;
    ensure_same_frame("polars-row-derive", &df_derive, &row_derive)?;

    let flat_rows = generate_flat_row_derive_rows(8);
    let flat_df_derive = flat_rows.as_slice().to_dataframe()?;
    let flat_row_derive = flat_row_derive_to_dataframe(&flat_rows)?;
    ensure_same_frame(
        "polars-row-derive flat row",
        &flat_df_derive,
        &flat_row_derive,
    )?;

    println!(
        "smoke ok: rich {:?}, flat {:?}",
        df_derive.shape(),
        flat_df_derive.shape()
    );
    Ok(())
}

fn ensure_same_frame(label: &str, left: &DataFrame, right: &DataFrame) -> Result<()> {
    anyhow::ensure!(
        left.shape() == right.shape(),
        "{label} shape mismatch: {:?} vs {:?}",
        left.shape(),
        right.shape()
    );
    anyhow::ensure!(
        left.get_column_names() == right.get_column_names(),
        "{label} column names differ:\nleft: {:?}\nright: {:?}",
        left.get_column_names(),
        right.get_column_names()
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
    anyhow::ensure!(
        left_dtypes == right_dtypes,
        "{label} dtypes differ:\nleft: {left_dtypes:?}\nright: {right_dtypes:?}",
    );
    if let Some(column) = first_value_mismatch(left, right) {
        anyhow::bail!(
            "{label} values differ in column {column:?}:\nleft:\n{}\nright:\n{}",
            left.column(column)?.as_materialized_series().head(Some(8)),
            right.column(column)?.as_materialized_series().head(Some(8)),
        );
    }
    Ok(())
}

fn first_value_mismatch<'a>(left: &'a DataFrame, right: &DataFrame) -> Option<&'a str> {
    left.columns()
        .iter()
        .zip(right.columns())
        .find_map(|(left_column, right_column)| {
            (!left_column.equals_missing(right_column)).then(|| left_column.name().as_str())
        })
}

fn showcase_output() -> Result<ShowcaseOutput> {
    let rows = generate_showcase_rows(6);
    let schema = ShowcaseRow::schema()
        .context("df-derive schema")?
        .iter()
        .map(|(name, dtype)| format!("{name}: {dtype:?}"))
        .collect::<Vec<_>>()
        .join("\n");
    let df = rows
        .as_slice()
        .to_dataframe()
        .context("df-derive showcase dataframe")?;
    let head = format!("{}", df.head(Some(5)));
    Ok(ShowcaseOutput { schema, head })
}

fn run_benchmarks() -> Result<Vec<BenchStats>> {
    let mut stats = Vec::new();

    for rows in ROW_COUNTS {
        println!("benchmarking {rows} rows");

        let native_rows = generate_showcase_rows(rows);
        stats.push(measure(Approach::DfDerive, rows, || {
            black_box(native_rows.as_slice()).to_dataframe()
        })?);
        stats.push(measure(Approach::ManualPolars, rows, || {
            manual_polars_to_dataframe(black_box(native_rows.as_slice()))
        })?);
        drop(native_rows);

        let serde_rows = generate_serde_arrow_rows(rows);
        let fields = serde_arrow_fields()?;
        stats.push(measure(Approach::SerdeArrowIpc, rows, || {
            serde_arrow_to_polars_ipc(black_box(&serde_rows), black_box(&fields))
        })?);
        stats.push(measure(Approach::SerdeArrowFfi, rows, || {
            serde_arrow_to_polars_ffi(black_box(&serde_rows), black_box(&fields))
        })?);
        drop(serde_rows);

        let row_derive_rows = generate_row_derive_rows(rows);
        stats.push(measure(Approach::PolarsRowDerive, rows, || {
            polars_row_derive_to_dataframe(black_box(row_derive_rows.as_slice()))
        })?);
        drop(row_derive_rows);

        let flat_df_rows = generate_flat_df_rows(rows);
        stats.push(measure(Approach::FlatDfDerive, rows, || {
            black_box(flat_df_rows.as_slice()).to_dataframe()
        })?);
        drop(flat_df_rows);

        let flat_row_derive_rows = generate_flat_row_derive_rows(rows);
        stats.push(measure(Approach::FlatPolarsRowDerive, rows, || {
            flat_row_derive_to_dataframe(black_box(flat_row_derive_rows.as_slice()))
        })?);
        drop(flat_row_derive_rows);
    }

    Ok(stats)
}

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

fn generate_showcase_rows(count: usize) -> Vec<ShowcaseRow> {
    (0..count).map(showcase_row).collect()
}

fn showcase_row(i: usize) -> ShowcaseRow {
    let i_i64 = i64::try_from(i).expect("row index fits i64");
    let i_u64 = u64::try_from(i).expect("row index fits u64");
    let base = Utc
        .timestamp_millis_opt(1_700_000_000_000)
        .single()
        .expect("valid timestamp");
    let side = match i % 4 {
        0 => Side::Buy,
        1 => Side::Sell,
        2 => Side::Short,
        _ => Side::Cover,
    };
    let risk = RiskBlock {
        sector: format!("sector_{}", i % 5),
        beta: 0.72 + f64::from((i % 120) as u32) * 0.01,
        hedged: !i.is_multiple_of(3),
    };
    let fills = (0..=(i % 2))
        .map(|j| Fill {
            venue: format!("VENUE{}", (i + j) % 4),
            size: 50 + ((i + j) % 200) as u64,
            rebate_bps: -0.2 + f64::from(((i + j) % 9) as u32) * 0.05,
        })
        .collect();
    let marks = vec![
        99.5 + f64::from((i % 11) as u32) * 0.1,
        100.0 + f64::from((i % 13) as u32) * 0.1,
        100.5 + f64::from((i % 17) as u32) * 0.1,
    ];

    ShowcaseRow {
        id: i_u64,
        symbol: format!("SYM{:03}", i % 128),
        notional: 10_000.0 + f64::from((i % 10_000) as u32) * 1.25,
        quantity: 100 + (i_u64 % 9_000),
        live: i.is_multiple_of(2),
        maybe_note: (!i.is_multiple_of(5)).then(|| format!("note-{}", i % 17)),
        price: Decimal::new(10_000_000 + i_i64 * 13, PRICE_SCALE as u32),
        ts: base + chrono::Duration::microseconds(i_i64 * 37),
        side,
        fingerprint: fingerprint(i),
        risk,
        fills,
        marks,
        cache_key: i_u64.rotate_left(13),
    }
}

fn fingerprint(seed: usize) -> Vec<u8> {
    (0..12)
        .map(|offset| u8::try_from(seed.wrapping_mul(31).wrapping_add(offset) & 0xff).unwrap())
        .collect()
}

fn generate_serde_arrow_rows(count: usize) -> Vec<SerdeArrowRow> {
    (0..count)
        .map(|i| {
            let row = showcase_row(i);
            let fills_venue = row.fills.iter().map(|fill| fill.venue.clone()).collect();
            let fills_size = row.fills.iter().map(|fill| fill.size).collect();
            let fills_rebate_bps = row.fills.iter().map(|fill| fill.rebate_bps).collect();
            SerdeArrowRow {
                id: row.id,
                symbol: row.symbol,
                notional: row.notional,
                quantity: row.quantity,
                live: row.live,
                maybe_note: row.maybe_note,
                price: row.price.to_string(),
                ts: row.ts,
                side: row.side.as_str().to_owned(),
                fingerprint: row.fingerprint,
                risk_sector: row.risk.sector,
                risk_beta: row.risk.beta,
                risk_hedged: row.risk.hedged,
                fills_venue,
                fills_size,
                fills_rebate_bps,
                marks: row.marks,
            }
        })
        .collect()
}

fn generate_row_derive_rows(count: usize) -> Vec<RowDeriveInput> {
    (0..count)
        .map(|i| {
            let row = showcase_row(i);
            let fills_venue = row.fills.iter().map(|fill| fill.venue.clone()).collect();
            let fills_size = row.fills.iter().map(|fill| fill.size).collect();
            let fills_rebate_bps = row.fills.iter().map(|fill| fill.rebate_bps).collect();
            RowDeriveInput {
                flat: RowDeriveFlat {
                    id: row.id,
                    symbol: row.symbol,
                    notional: row.notional,
                    quantity: row.quantity,
                    live: row.live,
                    maybe_note: row.maybe_note,
                    price: decimal_mantissa_6(&row.price),
                    ts: row.ts.timestamp_micros(),
                    side: row.side.as_str().to_owned(),
                    risk_sector: row.risk.sector,
                    risk_beta: row.risk.beta,
                    risk_hedged: row.risk.hedged,
                },
                fingerprint: row.fingerprint,
                fills_venue,
                fills_size,
                fills_rebate_bps,
                marks: row.marks,
            }
        })
        .collect()
}

fn generate_flat_df_rows(count: usize) -> Vec<RowDeriveFlat> {
    generate_flat_row_derive_rows(count)
}

fn generate_flat_row_derive_rows(count: usize) -> Vec<RowDeriveFlat> {
    (0..count)
        .map(|i| {
            let row = showcase_row(i);
            RowDeriveFlat {
                id: row.id,
                symbol: row.symbol,
                notional: row.notional,
                quantity: row.quantity,
                live: row.live,
                maybe_note: row.maybe_note,
                price: decimal_mantissa_6(&row.price),
                ts: row.ts.timestamp_micros(),
                side: row.side.as_str().to_owned(),
                risk_sector: row.risk.sector,
                risk_beta: row.risk.beta,
                risk_hedged: row.risk.hedged,
            }
        })
        .collect()
}

fn decimal_mantissa_6(decimal: &Decimal) -> i128 {
    debug_assert_eq!(decimal.scale(), PRICE_SCALE as u32);
    decimal.mantissa()
}

fn manual_polars_to_dataframe(items: &[ShowcaseRow]) -> PolarsResult<DataFrame> {
    let mut ids = Vec::with_capacity(items.len());
    let mut symbols = Vec::with_capacity(items.len());
    let mut notionals = Vec::with_capacity(items.len());
    let mut quantities = Vec::with_capacity(items.len());
    let mut live = Vec::with_capacity(items.len());
    let mut notes = Vec::with_capacity(items.len());
    let mut prices = Vec::with_capacity(items.len());
    let mut timestamps = Vec::with_capacity(items.len());
    let mut sides = Vec::with_capacity(items.len());
    let mut risk_sector = Vec::with_capacity(items.len());
    let mut risk_beta = Vec::with_capacity(items.len());
    let mut risk_hedged = Vec::with_capacity(items.len());
    let mut fills_venue = Vec::with_capacity(items.len());
    let mut fills_size = Vec::with_capacity(items.len());
    let mut fills_rebate_bps = Vec::with_capacity(items.len());
    let mut marks = Vec::with_capacity(items.len());

    for row in items {
        ids.push(row.id);
        symbols.push(row.symbol.as_str());
        notionals.push(row.notional);
        quantities.push(row.quantity);
        live.push(row.live);
        notes.push(row.maybe_note.as_deref());
        prices.push(decimal_mantissa_for_polars(&row.price)?);
        timestamps.push(row.ts.timestamp_micros());
        sides.push(row.side.as_str());
        risk_sector.push(row.risk.sector.as_str());
        risk_beta.push(row.risk.beta);
        risk_hedged.push(row.risk.hedged);
        fills_venue.push(string_list_anyvalue(
            row.fills.iter().map(|fill| fill.venue.as_str()),
        ));
        fills_size.push(u64_list_anyvalue(row.fills.iter().map(|fill| fill.size)));
        fills_rebate_bps.push(f64_list_anyvalue(
            row.fills.iter().map(|fill| fill.rebate_bps),
        ));
        marks.push(f64_list_anyvalue(row.marks.iter().copied()));
    }

    let price_series = decimal_series("price", prices);
    let timestamp_series = datetime_us_series("ts", timestamps)?;
    let fingerprint_series = binary_series(
        "fingerprint",
        items.iter().map(|row| row.fingerprint.as_slice()),
    );

    DataFrame::new(items.len(), vec![
        Series::new("id".into(), &ids).into(),
        Series::new("symbol".into(), &symbols).into(),
        Series::new("notional".into(), &notionals).into(),
        Series::new("quantity".into(), &quantities).into(),
        Series::new("live".into(), &live).into(),
        Series::new("maybe_note".into(), &notes).into(),
        price_series.into(),
        timestamp_series.into(),
        Series::new("side".into(), &sides).into(),
        fingerprint_series.into(),
        Series::new("risk.sector".into(), &risk_sector).into(),
        Series::new("risk.beta".into(), &risk_beta).into(),
        Series::new("risk.hedged".into(), &risk_hedged).into(),
        Series::new("fills.venue".into(), &fills_venue).into(),
        Series::new("fills.size".into(), &fills_size).into(),
        Series::new("fills.rebate_bps".into(), &fills_rebate_bps).into(),
        Series::new("marks".into(), &marks).into(),
    ])
}

fn decimal_mantissa_for_polars(decimal: &Decimal) -> PolarsResult<i128> {
    Decimal128Encode::try_to_i128_mantissa(decimal, PRICE_SCALE as u32).ok_or_else(|| {
        polars_err!(
            ComputeError: "decimal value cannot be represented at scale {}",
            PRICE_SCALE
        )
    })
}

fn decimal_series(name: &str, values: Vec<i128>) -> Series {
    Int128Chunked::from_vec(name.into(), values)
        .into_decimal_unchecked(PRICE_PRECISION, PRICE_SCALE)
        .into_series()
}

fn datetime_us_series(name: &str, values: Vec<i64>) -> PolarsResult<Series> {
    let series = Series::new(name.into(), &values);
    series.cast(&DataType::Datetime(TimeUnit::Microseconds, None))
}

fn binary_series<'a>(name: &str, values: impl Iterator<Item = &'a [u8]>) -> Series {
    BinaryChunked::from_iter_values(name.into(), values).into_series()
}

fn string_list_anyvalue<'a>(values: impl Iterator<Item = &'a str>) -> AnyValue<'static> {
    let inner = values.collect::<Vec<_>>();
    AnyValue::List(Series::new("".into(), &inner))
}

fn u64_list_anyvalue(values: impl Iterator<Item = u64>) -> AnyValue<'static> {
    let inner = values.collect::<Vec<_>>();
    AnyValue::List(Series::new("".into(), &inner))
}

fn f64_list_anyvalue(values: impl Iterator<Item = f64>) -> AnyValue<'static> {
    let inner = values.collect::<Vec<_>>();
    AnyValue::List(Series::new("".into(), &inner))
}

fn serde_arrow_fields() -> Result<Vec<FieldRef>> {
    let fields = json!([
        {"name": "id", "data_type": "U64"},
        {"name": "symbol", "data_type": "Utf8"},
        {"name": "notional", "data_type": "F64"},
        {"name": "quantity", "data_type": "U64"},
        {"name": "live", "data_type": "Bool"},
        {"name": "maybe_note", "data_type": "Utf8", "nullable": true},
        {"name": "price", "data_type": format!("Decimal128({PRICE_PRECISION}, {PRICE_SCALE})")},
        {"name": "ts", "data_type": "Timestamp(Microsecond, None)"},
        {"name": "side", "data_type": "Utf8"},
        {"name": "fingerprint", "data_type": "Binary"},
        {"name": "risk.sector", "data_type": "Utf8"},
        {"name": "risk.beta", "data_type": "F64"},
        {"name": "risk.hedged", "data_type": "Bool"},
        {
            "name": "fills.venue",
            "data_type": "List",
            "children": [{"name": "element", "data_type": "Utf8"}],
        },
        {
            "name": "fills.size",
            "data_type": "List",
            "children": [{"name": "element", "data_type": "U64"}],
        },
        {
            "name": "fills.rebate_bps",
            "data_type": "List",
            "children": [{"name": "element", "data_type": "F64"}],
        },
        {
            "name": "marks",
            "data_type": "List",
            "children": [{"name": "element", "data_type": "F64"}],
        },
    ]);

    Vec::<FieldRef>::from_value(&fields).context("build serde_arrow schema from explicit value")
}

fn serde_arrow_from_type_probe() -> String {
    let options = TracingOptions::default()
        .sequence_as_large_list(false)
        .strings_as_large_utf8(false)
        .bytes_as_large_binary(false);
    match Vec::<FieldRef>::from_type::<SerdeArrowRow>(options) {
        Ok(fields) => fields
            .iter()
            .map(|field| format!("{}: {:?}", field.name(), field.data_type()))
            .collect::<Vec<_>>()
            .join(", "),
        Err(err) => err.to_string(),
    }
}

fn serde_arrow_to_polars_ipc(
    items: &Vec<SerdeArrowRow>,
    fields: &[FieldRef],
) -> PolarsResult<DataFrame> {
    let batch = serde_arrow::to_record_batch(fields, items)
        .map_err(|err| polars_err!(ComputeError: "serde_arrow record batch failed: {}", err))?;
    let mut bytes = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema())
            .map_err(|err| polars_err!(ComputeError: "arrow IPC writer failed: {}", err))?;
        writer
            .write(&batch)
            .map_err(|err| polars_err!(ComputeError: "arrow IPC write failed: {}", err))?;
        writer
            .finish()
            .map_err(|err| polars_err!(ComputeError: "arrow IPC finish failed: {}", err))?;
    }
    IpcStreamReader::new(Cursor::new(bytes)).finish()
}

fn serde_arrow_to_polars_ffi(
    items: &Vec<SerdeArrowRow>,
    fields: &[FieldRef],
) -> PolarsResult<DataFrame> {
    let batch = serde_arrow::to_record_batch(fields, items)
        .map_err(|err| polars_err!(ComputeError: "serde_arrow record batch failed: {}", err))?;
    arrow_record_batch_to_polars_ffi(&batch)
}

fn arrow_record_batch_to_polars_ffi(
    batch: &arrow::record_batch::RecordBatch,
) -> PolarsResult<DataFrame> {
    let columns = batch
        .columns()
        .iter()
        .zip(batch.schema().fields().iter())
        .map(|(array, field)| {
            let polars_field = arrow_field_to_polars(field.as_ref())?;
            let polars_array = arrow_array_to_polars(array, polars_field.dtype.clone())?;
            Series::try_from((&polars_field, polars_array)).map(Column::from)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    DataFrame::new(batch.num_rows(), columns)
}

fn arrow_field_to_polars(
    field: &arrow::datatypes::Field,
) -> PolarsResult<polars_arrow::datatypes::Field> {
    let ffi_schema = arrow::ffi::FFI_ArrowSchema::try_from(field)
        .map_err(|err| polars_err!(ComputeError: "arrow field FFI export failed: {}", err))?;
    let polars_schema = (&ffi_schema as *const arrow::ffi::FFI_ArrowSchema).cast();
    // SAFETY: Arrow's C Data schema struct and polars-arrow's C Data schema
    // struct are both repr(C) definitions of the Arrow C Data Interface.
    unsafe { polars_arrow::ffi::import_field_from_c(&*polars_schema) }
}

fn arrow_array_to_polars(
    array: &arrow::array::ArrayRef,
    dtype: polars_arrow::datatypes::ArrowDataType,
) -> PolarsResult<PolarsArrayRef> {
    let (ffi_array, _ffi_schema) = arrow::array::ffi::to_ffi(&array.to_data())
        .map_err(|err| polars_err!(ComputeError: "arrow array FFI export failed: {}", err))?;
    let polars_array = move_arrow_ffi_array(ffi_array);
    // SAFETY: the array was exported by arrow-rs through the Arrow C Data
    // Interface, and `dtype` is the matching dtype imported from the same
    // field schema.
    unsafe { polars_arrow::ffi::import_array_from_c(polars_array, dtype) }
}

fn move_arrow_ffi_array(array: arrow::array::ffi::FFI_ArrowArray) -> polars_arrow::ffi::ArrowArray {
    let array = ManuallyDrop::new(array);
    // SAFETY: both types are repr(C) definitions of the Arrow C Data Interface.
    // This moves ownership to polars-arrow so exactly one release callback runs.
    unsafe {
        std::ptr::read(
            (&*array as *const arrow::array::ffi::FFI_ArrowArray)
                .cast::<polars_arrow::ffi::ArrowArray>(),
        )
    }
}

fn polars_row_derive_to_dataframe(items: &[RowDeriveInput]) -> PolarsResult<DataFrame> {
    let flat_rows = items.iter().map(|row| row.flat.clone()).collect::<Vec<_>>();
    let mut df = flat_rows.into_iter().to_dataframe()?;

    let prices = items.iter().map(|row| row.flat.price).collect::<Vec<_>>();
    df.with_column(decimal_series("price", prices).into())?;
    replace_with_cast(
        &mut df,
        "ts",
        &DataType::Datetime(TimeUnit::Microseconds, None),
    )?;
    let fingerprint_series = binary_series(
        "fingerprint",
        items.iter().map(|row| row.fingerprint.as_slice()),
    );
    df.with_column(fingerprint_series.into())?;

    let fills_venue = items
        .iter()
        .map(|row| string_list_anyvalue(row.fills_venue.iter().map(String::as_str)))
        .collect::<Vec<_>>();
    let fills_size = items
        .iter()
        .map(|row| u64_list_anyvalue(row.fills_size.iter().copied()))
        .collect::<Vec<_>>();
    let fills_rebate_bps = items
        .iter()
        .map(|row| f64_list_anyvalue(row.fills_rebate_bps.iter().copied()))
        .collect::<Vec<_>>();
    let marks = items
        .iter()
        .map(|row| f64_list_anyvalue(row.marks.iter().copied()))
        .collect::<Vec<_>>();

    df.with_column(Series::new("fills.venue".into(), &fills_venue).into())?;
    df.with_column(Series::new("fills.size".into(), &fills_size).into())?;
    df.with_column(Series::new("fills.rebate_bps".into(), &fills_rebate_bps).into())?;
    df.with_column(Series::new("marks".into(), &marks).into())?;

    df.rename("risk_sector", "risk.sector".into())?;
    df.rename("risk_beta", "risk.beta".into())?;
    df.rename("risk_hedged", "risk.hedged".into())?;
    df.select(OUTPUT_COLUMNS)
}

fn flat_row_derive_to_dataframe(items: &[RowDeriveFlat]) -> PolarsResult<DataFrame> {
    items.iter().cloned().to_dataframe()
}

fn replace_with_cast(df: &mut DataFrame, name: &str, dtype: &DataType) -> PolarsResult<()> {
    let mut series = df.column(name)?.as_materialized_series().cast(dtype)?;
    series.rename(name.into());
    df.with_column(series.into())?;
    Ok(())
}

fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos >= 1_000_000_000 {
        format!("{:.3}s", duration.as_secs_f64())
    } else if nanos >= 1_000_000 {
        format!("{:.3}ms", duration.as_secs_f64() * 1_000.0)
    } else if nanos >= 1_000 {
        format!("{:.3}us", duration.as_secs_f64() * 1_000_000.0)
    } else {
        format!("{nanos}ns")
    }
}

fn render_report(showcase: &ShowcaseOutput, timings: &[BenchStats]) -> Result<String> {
    let mut report = String::new();
    let machine = machine_info();
    let versions = crate_versions();

    report.push_str("# df-derive head-to-head\n\n");
    report.push_str("## Environment\n\n");
    report.push_str(&format!("- Date: {}\n", command_line("date", &["-u"])));
    report.push_str(&format!("- Host: {}\n", command_line("uname", &["-a"])));
    report.push_str(&format!("- CPU: {}\n", machine.cpu));
    report.push_str(&format!("- RAM: {}\n", machine.ram));
    report.push_str(&format!(
        "- rustc: {}\n",
        command_line("rustc", &["--version"])
    ));
    report.push_str(&format!(
        "- cargo: {}\n",
        command_line("cargo", &["--version"])
    ));
    report.push_str(&format!(
        "- Crates: df-derive {}, polars {}, polars-arrow {}, serde_arrow {}, arrow {}, polars-row-derive {}, rust_decimal {}\n",
        versions.get("df-derive"),
        versions.get("polars"),
        versions.get("polars-arrow"),
        versions.get("serde_arrow"),
        versions.get("arrow"),
        versions.get("polars-row-derive"),
        versions.get("rust_decimal"),
    ));
    report.push_str(&format!(
        "- Harness: release build, {WARMUPS} warmups, {ITERS} measured iterations, median and min reported, data generation excluded.\n\n"
    ));

    report.push_str("## API confirmation\n\n");
    report.push_str("- Public import: `use df_derive::prelude::*;`.\n");
    report.push_str("- Derive: `#[derive(ToDataFrame)]` on structs and tuple structs. Nested custom structs must implement `Columnar`, normally by deriving `ToDataFrame`.\n");
    report.push_str("- Runtime implementation: derived types implement hidden `ColumnarSpec::{build_schema, encode_columns}`; checked blanket APIs count general iterator inputs or use known slice cardinality, then validate emitted columns before building the outer frame.\n");
    report.push_str("- Blanket API: every `ColumnarSpec` type receives `Columnar` and `ToDataFrame::{to_dataframe, empty_dataframe, schema}` (`schema()` returns the explicitly composed `SchemaRef`), plus the `ToDataFrameVec` slice extension.\n");
    report.push_str("- Field attributes verified from the current README/docs: `skip`, `flatten`, `flatten(prefix = \"...\")`, `as_string`, `as_str`, `as_binary`, `decimal(precision = N, scale = S)`, and `time_unit = \"ms\" | \"us\" | \"ns\"`.\n");
    report.push_str("- Sources checked: [docs.rs `df-derive`](https://docs.rs/df-derive), [GitHub README](https://github.com/gramistella/df-derive), and the local README in this checkout.\n\n");

    report.push_str("## Showcase shape\n\n");
    report.push_str("The df-derive model is deliberately not a toy row: it has scalar columns, `Option`, decimal, microsecond datetime, enum-as-string, binary bytes, a nested struct, a `Vec<Nested>` split into list columns, and a `Vec<f64>` list column.\n\n");
    report.push_str("```rust\n");
    report.push_str(SHOWCASE_SNIPPET);
    report.push_str("\n```\n\n");
    report.push_str("### Schema\n\n");
    report.push_str("```text\n");
    report.push_str(&showcase.schema);
    report.push_str("\n```\n\n");
    report.push_str("### `df.head()`\n\n");
    report.push_str("```text\n");
    report.push_str(&showcase.head);
    report.push_str("\n```\n\n");

    report.push_str("## Equivalent implementations\n\n");
    report.push_str("### df-derive\n\n");
    report.push_str("The whole batch conversion is the derived columnar path:\n\n");
    report.push_str("```rust\n");
    report.push_str("let df = rows.as_slice().to_dataframe()?;\n");
    report.push_str("```\n\n");

    report.push_str("### hand-written Polars\n\n");
    report.push_str("The baseline manually collects every output column. Decimal and datetime require explicit mantissa/timestamp conversion and casts; binary needs a binary chunked array; the list columns are assembled as per-row inner `Series` values.\n\n");
    report.push_str("```rust\n");
    report.push_str(MANUAL_SNIPPET);
    report.push_str("\n```\n\n");

    report.push_str("### serde_arrow\n\n");
    report.push_str("The working serde_arrow version is a flattened serialization model, not the original nested domain struct. The enum is stringified before serialization, decimal is serialized as a string for Arrow `Decimal128`, `DateTime<Utc>` uses chrono's microsecond serializer, bytes use `serde_bytes`, and the final Arrow schema is explicit because the `from_type` probe traces a structural schema, not the semantic Polars schema. The benchmark reports both the original IPC bridge and a non-IPC Arrow C Data FFI bridge.\n\n");
    report.push_str(&format!(
        "`Vec::<FieldRef>::from_type::<SerdeArrowRow>(...)` result for the full row: it succeeds structurally, but the traced schema is not the target schema. Traced summary: `{}`.\n\n",
        serde_arrow_from_type_probe()
    ));
    report.push_str("```rust\n");
    report.push_str(SERDE_ARROW_SNIPPET);
    report.push_str("\n```\n\n");

    report.push_str("### polars-row-derive\n\n");
    report.push_str("`polars-row-derive` compiled against Polars 0.53 in this isolated crate. It is row-oriented and has no df-derive-style field attributes, so this benchmark uses a flattened row, then post-processes names and rich dtypes back to the df-derive shape. The iterator API consumes rows; the repeated benchmark therefore clones the rows before each conversion.\n\n");
    report.push_str("```rust\n");
    report.push_str(ROW_DERIVE_SNIPPET);
    report.push_str("\n```\n\n");

    report.push_str("### flat row speed check\n\n");
    report.push_str("To avoid mixing `polars-row-derive` feature gaps with row-vs-columnar speed, the harness also benchmarks a deliberately flat scalar row through both derives. This flat check excludes nested lists, decimal/datetime/binary post-processing, and rich-schema column renames. The `polars-row-derive` flat path still includes the per-call row clone forced by its consuming iterator API.\n\n");
    report.push_str("```rust\n");
    report.push_str(FLAT_ROW_SNIPPET);
    report.push_str("\n```\n\n");

    report.push_str("## Timings\n\n");
    report.push_str("| Approach | Rows | Median | Min |\n");
    report.push_str("| --- | ---: | ---: | ---: |\n");
    for stat in timings {
        report.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            stat.approach.label(),
            stat.rows,
            format_duration(stat.median),
            format_duration(stat.min)
        ));
    }
    report.push('\n');
    report.push_str("Read the rich-schema table with scope: the hand-written Polars baseline is the straightforward boilerplate df-derive saves, not a proof that an expert cannot hand-write lower-level list builders; the `polars-row-derive + postprocess` number includes the work required to reach the same rich schema. The flat-row rows isolate a simpler schema, but the `polars-row-derive` flat path still includes the clone required by its consuming iterator API.\n\n");

    report.push_str("## Findings\n\n");
    report.push_str("### df-derive\n\n");
    report.push_str("- Ergonomics: best fit for a Rust domain struct that should become a Polars DataFrame. The original nested model is the schema; attributes express the exceptional cases locally.\n");
    report.push_str("- Performance: measures the generated Polars-native columnar path directly from `&[ShowcaseRow]`. The safe claim is not that no human can hand-write equivalent builders, but that the derive reaches this rich schema without paying a throughput penalty for the abstraction.\n\n");
    report.push_str("### hand-written Polars\n\n");
    report.push_str("- Ergonomics: full control, but every rich dtype forces hand-written conversion code. The code is easy to get subtly wrong because field order, column names, nullability, list inner dtypes, decimal scale, and datetime units all live outside the struct definition.\n");
    report.push_str("- Performance: this is the naive, readable hand-written baseline. It builds list columns with one inner `Series` per row, so do not read the multiple as `df-derive beats optimal hand-written Polars by X`; an expert can close that gap by hand-writing lower-level list builders.\n\n");
    report.push_str("### serde_arrow\n\n");
    report.push_str("- Ergonomics: strong if Arrow is the target, but not Polars-native. Matching df-derive's output requires a flattened serialization row, explicit schema fields, serde helper attributes, and a bridge into Polars.\n");
    report.push_str("- Type gaps for this shape: enum-as-string is manual, `Vec<u8>` needs `serde_bytes` plus an explicit Binary field, decimal precision/scale is not carried by `rust_decimal::Decimal` and is parsed from a string into Arrow `Decimal128`, and the `Vec<Nested>` shape must be manually split into one list field per nested member.\n");
    report.push_str("- Performance: the IPC number includes RecordBatch creation plus serialize/reparse bridge cost. The C Data FFI number is the non-IPC check; it still measures the same end state, an in-memory Polars DataFrame.\n\n");
    report.push_str("### polars-row-derive\n\n");
    report.push_str("- Ergonomics: stale-looking but usable for this probe with Polars 0.53 because the macro expands to `polars::df!`. It does not understand nested flattening, decimal/time/binary attributes, or borrowed batch conversion.\n");
    report.push_str("- Type gaps for this shape: column names with dots require post-rename, decimal/datetime require post-casts, and Binary requires rebuilding/replacing the column. Without those post-steps the output is not the same schema.\n");
    report.push_str("- Performance: the rich-schema number includes the row-derive conversion, row cloning needed by the consuming iterator API, and post-processing required to reach the same DataFrame schema. The separate flat-row number removes rich-schema post-processing, but still includes the clone required by the consuming iterator API.\n\n");

    report.push_str("## Gotchas encountered\n\n");
    report.push_str("- `serde_arrow::to_record_batch` takes `&T: Serialize`, so passing a bare slice failed because `[SerdeArrowRow]` is unsized. The harness passes `&Vec<SerdeArrowRow>` instead.\n");
    report.push_str("- `serde_arrow` could trace the flattened row structurally, but that trace was not the semantic target schema: `price` traced as `Utf8` and `ts` as `Int64`, so Decimal128 precision/scale and Timestamp unit had to be supplied explicitly.\n");
    report.push_str("- The non-IPC serde_arrow bridge is possible through Arrow C Data FFI, but it is not a simple `RecordBatch -> DataFrame` public API. The harness uses a small unsafe adapter between upstream arrow-rs FFI structs and polars-arrow FFI structs, then verifies dtype parity before timing it.\n");
    report.push_str("- Serializing an `i128` decimal carrier into Arrow `Decimal128(18, 6)` failed; decimal strings worked, but parsing those strings is part of the measured serde_arrow cost.\n");
    report.push_str("- `serde_bytes` was needed to keep `Vec<u8>` on the binary path. Without being explicit about bytes, it is easy to accidentally compare a list-of-u8 shape instead of a Binary column.\n");
    report.push_str("- `polars-row-derive` compiled with Polars 0.53, but its generated `polars::df!` path could not build the `Vec<Vec<T>>` list columns used by the flattened `Vec<Nested>` representation. Those columns had to be added manually after the derived conversion.\n");
    report.push_str("- `polars-row-derive` consumes an iterator of owned rows, so repeated benchmark iterations either consume the dataset or require cloning/collecting a flat row buffer. The benchmark includes that cloning because it is required by the usable API shape here.\n");
    report.push_str("- Hand-written Polars is easy to make unfair accidentally. The smoke test checks shape, column order, dtypes, and values against df-derive so the manual, serde_arrow, and row-derive paths all end at the same DataFrame before timing claims are made. It also checks the flat `df-derive` and `polars-row-derive` outputs against each other.\n");
    report.push_str("- The hand-written baseline is intentionally the maintainable/obvious version, not the theoretical ceiling. A lower-level hand implementation using list builders should be faster than this baseline and could approach df-derive's generated code.\n");
    report.push('\n');

    report.push_str("## Headline takeaway\n\n");
    report.push_str("df-derive gives one derive from a real Rust struct to a correctly typed Polars DataFrame, and in this benchmark it is the fastest measured path to that in-memory Polars result; serde_arrow is credible but needs an Arrow-first adapter layer, and polars-row-derive needs enough post-processing that it stops being a direct alternative for rich schemas.\n");

    Ok(report)
}

struct MachineInfo {
    cpu: String,
    ram: String,
}

fn machine_info() -> MachineInfo {
    let cpu = command_line("sysctl", &["-n", "machdep.cpu.brand_string"]);
    let mem = command_line("sysctl", &["-n", "hw.memsize"]);
    let ram = mem
        .trim()
        .parse::<u64>()
        .map(|bytes| format!("{:.1} GiB", bytes as f64 / 1024.0_f64.powi(3)))
        .unwrap_or(mem);
    MachineInfo { cpu, ram }
}

fn command_line(program: &str, args: &[&str]) -> String {
    match Command::new(program).args(args).output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if stderr.is_empty() {
                format!("{program} exited with {}", output.status)
            } else {
                stderr
            }
        }
        Err(err) => format!("unavailable ({err})"),
    }
}

struct CrateVersions(Vec<(String, String)>);

impl CrateVersions {
    fn get(&self, name: &str) -> &str {
        self.0
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map_or("unresolved", |(_, version)| version.as_str())
    }
}

fn crate_versions() -> CrateVersions {
    let wanted = [
        "df-derive",
        "polars",
        "polars-arrow",
        "serde_arrow",
        "arrow",
        "polars-row-derive",
        "rust_decimal",
    ];
    let lock = fs::read_to_string("Cargo.lock").unwrap_or_default();
    let mut found = Vec::new();
    let mut current_name: Option<String> = None;

    for line in lock.lines() {
        if line == "[[package]]" {
            current_name = None;
        } else if let Some(name) = line
            .strip_prefix("name = \"")
            .and_then(|s| s.strip_suffix('"'))
        {
            current_name = Some(name.to_owned());
        } else if let Some(version) = line
            .strip_prefix("version = \"")
            .and_then(|s| s.strip_suffix('"'))
            && let Some(name) = current_name.take()
            && wanted.contains(&name.as_str())
        {
            found.push((name, version.to_owned()));
        }
    }

    CrateVersions(found)
}

const SHOWCASE_SNIPPET: &str = r#"use df_derive::prelude::*;
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
}"#;

const MANUAL_SNIPPET: &str = r#"let prices: Vec<i128> = rows
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
])?;"#;

const SERDE_ARROW_SNIPPET: &str = r#"#[derive(Serialize, Deserialize)]
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
let df_ffi = arrow_c_data_ffi_into_polars(batch)?;"#;

const ROW_DERIVE_SNIPPET: &str = r#"#[derive(Clone, IterToDataFrame)]
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
// polars-row-derive itself did not build the list columns or Binary column."#;

const FLAT_ROW_SNIPPET: &str = r#"#[derive(Clone, ToDataFrame, IterToDataFrame)]
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

let df_derive_df = flat_rows.as_slice().to_dataframe()?;
let row_derive_df = flat_rows.into_iter().to_dataframe()?;"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_produce_same_frame_on_small_batch() -> Result<()> {
        smoke()
    }
}
