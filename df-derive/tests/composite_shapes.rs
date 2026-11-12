//! Row types that combine nested records, lists of records and maps.
//!
//! Each test takes a realistic composite (fund holdings, filings with their
//! exhibits, articles with thumbnails, a market summary keyed by exchange)
//! and checks three things: the derive accepts the type, the columns come out
//! as stated here, and the block encoder agrees with pushing one value at a
//! time.
//!
//! Where the natural table has columns that are only known at run time (one
//! per period, one per ticker), the model here is the long form: one row per
//! cell, or a map field that one `explode` turns into rows.

// The sample data is built from loop counters: casts between integer types,
// and "every third row has none" written as a remainder.
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::manual_is_multiple_of,
    clippy::missing_const_for_fn,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::fmt;

use chrono::{DateTime, NaiveDate, TimeZone as _, Utc};
use df_derive::ToDataFrame;
use df_derive::dataframe::builders::StrBuilder;
use df_derive::dataframe::{Columnar, ToDataFrame, ToDataFrameVec};
use polars::prelude::*;
use rust_decimal::Decimal;

#[path = "support/agreement.rs"]
mod agreement;
use agreement::check;

// --- building blocks -----------------------------------------------------------

/// A string-like enum. It names its encoding once, so fields, lists and map
/// keys of it need no attribute.
macro_rules! labelled {
    ($name:ident { $($variant:ident => $label:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum $name {
            $($variant),+
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                match self {
                    $(Self::$variant => $label),+
                }
            }
        }

        impl Columnar for $name {
            type Builder = StrBuilder;
        }
    };
}

labelled!(Currency { Usd => "USD", Eur => "EUR" });
labelled!(Sector { Technology => "technology", Healthcare => "healthcare", RealEstate => "realestate" });
labelled!(LineItem { TotalRevenue => "TotalRevenue", NetIncome => "NetIncome", FreeCashFlow => "FreeCashFlow" });
labelled!(Frequency { Annual => "annual", Quarterly => "quarterly", Trailing => "trailing" });
labelled!(ContentType { Story => "STORY", Video => "VIDEO" });
labelled!(MarketState { Pre => "PRE", Regular => "REGULAR", Closed => "CLOSED" });

#[derive(ToDataFrame, Clone)]
struct Money {
    amount: Decimal,
    currency: Currency,
}

/// An amount in cents. Odd amounts are in euros, so both currencies occur.
fn money(cents: i64) -> Money {
    Money {
        amount: Decimal::new(cents, 2),
        currency: if cents % 2 == 0 {
            Currency::Usd
        } else {
            Currency::Eur
        },
    }
}

fn at(seconds: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(1_700_000_000 + seconds, 0)
        .single()
        .unwrap()
}

fn day(offset: i32) -> NaiveDate {
    NaiveDate::from_num_days_from_ce_opt(738_000 + offset).unwrap()
}

// --- dtypes, spelled once ------------------------------------------------------

fn instant() -> DataType {
    DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))
}

fn list(inner: DataType) -> DataType {
    DataType::List(Box::new(inner))
}

const TEXT: DataType = DataType::String;
const REAL: DataType = DataType::Float64;
const DECIMAL: DataType = DataType::Decimal(38, 10);

#[track_caller]
fn assert_schema<T: ToDataFrame>(expected: &[(&str, DataType)]) {
    let expected: Vec<(String, DataType)> = expected
        .iter()
        .map(|(name, dtype)| ((*name).to_owned(), dtype.clone()))
        .collect();
    assert_eq!(T::schema().unwrap(), expected);
    // The frame agrees with the schema, also when it is empty.
    let empty = T::empty_dataframe().unwrap();
    let columns: Vec<(String, DataType)> = empty
        .columns()
        .iter()
        .map(|column| (column.name().to_string(), column.dtype().clone()))
        .collect();
    assert_eq!(columns, expected);
}

/// The values of a list cell, as strings.
fn strings(df: &DataFrame, column: &str, row: usize) -> Vec<String> {
    let AnyValue::List(values) = df.column(column).unwrap().get(row).unwrap() else {
        panic!("`{column}` is not a list column");
    };
    values
        .str()
        .unwrap()
        .iter()
        .map(|value| value.unwrap_or("<null>").to_owned())
        .collect()
}

// --- earnings dates: a flat table with zoned instants ---------------------------

#[derive(ToDataFrame, Clone)]
struct EarningsDate {
    symbol: String,
    #[df_derive(time_unit = "ns")]
    at: DateTime<Utc>,
    eps_estimate: Option<Decimal>,
    eps_reported: Option<Decimal>,
    surprise_percent: Option<f64>,
}

#[test]
fn earnings_dates_are_a_flat_table_with_zoned_instants() {
    assert_schema::<EarningsDate>(&[
        ("symbol", TEXT),
        ("at", instant()),
        ("eps_estimate", DECIMAL),
        ("eps_reported", DECIMAL),
        ("surprise_percent", REAL),
    ]);

    let rows: Vec<EarningsDate> = (0..300)
        .map(|index| EarningsDate {
            symbol: "AAPL".into(),
            at: at(index * 7_776_000),
            eps_estimate: Some(Decimal::new(150 + index, 2)),
            // The most recent date has not been reported yet.
            eps_reported: (index != 299).then(|| Decimal::new(155 + index, 2)),
            surprise_percent: (index != 299).then_some(3.3),
        })
        .collect();
    check(&rows);

    let df = rows.to_dataframe().unwrap();
    let instants = df.column("at").unwrap().datetime().unwrap();
    assert_eq!(
        instants.physical().get(1),
        Some(rows[1].at.timestamp_nanos_opt().unwrap())
    );
    assert_eq!(df.column("eps_reported").unwrap().null_count(), 1);
}

// --- statements: columns that are dates become rows -----------------------------

/// One cell of a financial statement. As a table it would have a column per
/// period; here a period is a value, so any number of them fits one schema.
#[derive(ToDataFrame, Clone)]
struct StatementValue {
    symbol: String,
    frequency: Frequency,
    period_end: NaiveDate,
    item: LineItem,
    value: Option<Money>,
}

/// The same data grouped by line item, with its values keyed by period.
#[derive(ToDataFrame, Clone)]
struct LineItemSeries {
    item: LineItem,
    values: BTreeMap<NaiveDate, Decimal>,
}

#[test]
fn statements_are_rows_per_cell_or_a_map_per_line_item() {
    assert_schema::<StatementValue>(&[
        ("symbol", TEXT),
        ("frequency", TEXT),
        ("period_end", DataType::Date),
        ("item", TEXT),
        ("value.amount", DECIMAL),
        ("value.currency", TEXT),
    ]);
    assert_schema::<LineItemSeries>(&[
        ("item", TEXT),
        ("values.key", list(DataType::Date)),
        ("values.value", list(DECIMAL)),
    ]);

    let items = [
        LineItem::TotalRevenue,
        LineItem::NetIncome,
        LineItem::FreeCashFlow,
    ];
    let cells: Vec<StatementValue> = (0..360)
        .map(|index| StatementValue {
            symbol: "MSFT".into(),
            frequency: match index % 5 {
                0 => Frequency::Trailing,
                1 => Frequency::Annual,
                _ => Frequency::Quarterly,
            },
            period_end: day(index / 3 * 91),
            item: items[index as usize % 3],
            // Not every cell exists; a missing one is a null amount and currency.
            value: (index % 7 != 0).then(|| money(i64::from(index) * 1_000)),
        })
        .collect();
    check(&cells);

    let series: Vec<LineItemSeries> = items
        .iter()
        .enumerate()
        .map(|(row, &item)| LineItemSeries {
            item,
            values: (0..4 * row as i32)
                .map(|period| (day(period * 91), Decimal::new(i64::from(period) * 100, 2)))
                .collect(),
        })
        .collect();
    check(&series);

    // A map field is two list columns, and one `explode` makes them rows:
    // the long table again, with a row for every period a line item has.
    let long = series
        .to_dataframe()
        .unwrap()
        .explode(
            ["values.key", "values.value"],
            ExplodeOptions {
                empty_as_null: false,
                keep_nulls: false,
            },
        )
        .unwrap();
    assert_eq!(long.height(), 4 + 8);
    assert_eq!(long.column("values.key").unwrap().dtype(), &DataType::Date);
    assert_eq!(long.column("values.value").unwrap().dtype(), &DECIMAL);
}

// --- fund data: nested records, lists of records and maps in one row -------------

#[derive(ToDataFrame, Clone)]
struct FundOverview {
    family: Option<String>,
    category: Option<String>,
}

#[derive(ToDataFrame, Clone)]
struct AssetClasses {
    cash: Option<f64>,
    stock: Option<f64>,
    bond: Option<f64>,
    other: Option<f64>,
}

#[derive(ToDataFrame, Clone)]
struct Holding {
    symbol: String,
    name: String,
    weight: f64,
}

/// A metric of a fund, with the fund's own figure beside its category's
/// average.
#[derive(ToDataFrame, Clone)]
struct FundMetric {
    metric: String,
    fund: Option<f64>,
    category_average: Option<f64>,
}

#[derive(ToDataFrame, Clone)]
struct FundData {
    symbol: String,
    description: Option<String>,
    overview: FundOverview,
    asset_classes: AssetClasses,
    top_holdings: Vec<Holding>,
    operations: Vec<FundMetric>,
    /// Keyed by rating names that are only known at run time.
    bond_ratings: BTreeMap<String, f64>,
    /// Keyed by an enum, with weights at a precision of their own: the
    /// attribute applies to the map's values.
    #[df_derive(decimal(precision = 9, scale = 6))]
    sector_weightings: BTreeMap<Sector, Decimal>,
}

fn fund(index: usize) -> FundData {
    let sectors = [Sector::Technology, Sector::Healthcare, Sector::RealEstate];
    FundData {
        symbol: format!("FUND{index}"),
        description: (index % 3 != 0).then(|| format!("An index fund, number {index}.")),
        overview: FundOverview {
            family: Some("Vanguard".into()),
            category: (index % 2 == 0).then(|| "Large Blend".into()),
        },
        asset_classes: AssetClasses {
            cash: Some(0.01),
            stock: Some(0.98),
            bond: None,
            other: (index % 4 == 0).then_some(0.01),
        },
        top_holdings: (0..index % 11)
            .map(|rank| Holding {
                symbol: format!("H{rank}"),
                name: format!("Holding number {rank}"),
                weight: 0.07 / (rank + 1) as f64,
            })
            .collect(),
        operations: vec![
            FundMetric {
                metric: "Annual Report Expense Ratio".into(),
                fund: Some(0.0003),
                category_average: Some(0.0078),
            },
            FundMetric {
                metric: "Total Net Assets".into(),
                fund: Some(1.3e12),
                category_average: None,
            },
        ],
        bond_ratings: if index % 2 == 0 {
            BTreeMap::new()
        } else {
            BTreeMap::from([("aaa".to_owned(), 0.6), ("bbb".to_owned(), 0.4)])
        },
        sector_weightings: sectors
            .iter()
            .take(index % 4)
            .map(|&sector| (sector, Decimal::new(250_000, 6)))
            .collect(),
    }
}

#[test]
fn fund_data_holds_records_lists_and_maps_in_one_row() {
    assert_schema::<FundData>(&[
        ("symbol", TEXT),
        ("description", TEXT),
        ("overview.family", TEXT),
        ("overview.category", TEXT),
        ("asset_classes.cash", REAL),
        ("asset_classes.stock", REAL),
        ("asset_classes.bond", REAL),
        ("asset_classes.other", REAL),
        ("top_holdings.symbol", list(TEXT)),
        ("top_holdings.name", list(TEXT)),
        ("top_holdings.weight", list(REAL)),
        ("operations.metric", list(TEXT)),
        ("operations.fund", list(REAL)),
        ("operations.category_average", list(REAL)),
        ("bond_ratings.key", list(TEXT)),
        ("bond_ratings.value", list(REAL)),
        ("sector_weightings.key", list(TEXT)),
        ("sector_weightings.value", list(DataType::Decimal(9, 6))),
    ]);

    let rows: Vec<FundData> = (0..150).map(fund).collect();
    check(&rows);

    let df = rows.to_dataframe().unwrap();
    assert_eq!(strings(&df, "bond_ratings.key", 1), ["aaa", "bbb"]);
    assert_eq!(
        strings(&df, "sector_weightings.key", 3),
        ["technology", "healthcare", "realestate"]
    );
    let AnyValue::List(weights) = df
        .column("sector_weightings.value")
        .unwrap()
        .get(3)
        .unwrap()
    else {
        panic!("expected a list");
    };
    assert_eq!(weights.get(0).unwrap(), AnyValue::Decimal(250_000, 9, 6));
}

// --- filings: a map inside a list of records -------------------------------------

#[derive(ToDataFrame, Clone)]
struct Filing {
    filed: NaiveDate,
    form: String,
    title: String,
    url: Option<String>,
    /// Exhibit type to document address, as many as the filing has.
    exhibits: BTreeMap<String, String>,
}

#[derive(ToDataFrame, Clone)]
struct Filings {
    symbol: String,
    filings: Vec<Filing>,
}

fn filing(index: usize) -> Filing {
    Filing {
        filed: day(index as i32),
        form: if index % 4 == 0 { "10-K" } else { "10-Q" }.into(),
        title: format!("Periodic report {index}"),
        url: (index % 5 != 0).then(|| format!("https://example.test/filing/{index}")),
        exhibits: (0..index % 4)
            .map(|exhibit| {
                (
                    format!("EX-{exhibit}"),
                    format!("https://example.test/filing/{index}/ex{exhibit}"),
                )
            })
            .collect(),
    }
}

#[test]
fn filings_keep_a_map_inside_a_list_of_records() {
    assert_schema::<Filing>(&[
        ("filed", DataType::Date),
        ("form", TEXT),
        ("title", TEXT),
        ("url", TEXT),
        ("exhibits.key", list(TEXT)),
        ("exhibits.value", list(TEXT)),
    ]);
    assert_schema::<Filings>(&[
        ("symbol", TEXT),
        ("filings.filed", list(DataType::Date)),
        ("filings.form", list(TEXT)),
        ("filings.title", list(TEXT)),
        ("filings.url", list(TEXT)),
        ("filings.exhibits.key", list(list(TEXT))),
        ("filings.exhibits.value", list(list(TEXT))),
    ]);

    let flat: Vec<Filing> = (0..300).map(filing).collect();
    check(&flat);

    let nested: Vec<Filings> = (0..120)
        .map(|company| Filings {
            symbol: format!("C{company}"),
            filings: (0..company % 9)
                .map(|index| filing(company + index))
                .collect(),
        })
        .collect();
    check(&nested);

    // Company 7 has seven filings: numbers 7 to 13. The third is filing 9,
    // which has one exhibit.
    let df = nested.to_dataframe().unwrap();
    let AnyValue::List(per_filing) = df.column("filings.exhibits.key").unwrap().get(7).unwrap()
    else {
        panic!("expected a list of lists");
    };
    assert_eq!(per_filing.len(), 7);
    let AnyValue::List(third) = per_filing.get(2).unwrap() else {
        panic!("expected a list");
    };
    let third: Vec<_> = third.str().unwrap().iter().flatten().collect();
    assert_eq!(third, ["EX-0"]);
}

// --- news: nested lists of records ----------------------------------------------

#[derive(ToDataFrame, Clone)]
struct Thumbnail {
    url: String,
    width: u32,
    height: u32,
    tag: String,
}

#[derive(ToDataFrame, Clone)]
struct Provider {
    name: String,
    url: Option<String>,
}

#[derive(ToDataFrame, Clone)]
struct Article {
    id: String,
    title: String,
    summary: Option<String>,
    #[df_derive(time_unit = "ns")]
    published_at: DateTime<Utc>,
    content_type: ContentType,
    provider: Option<Provider>,
    url: Option<String>,
    thumbnails: Vec<Thumbnail>,
    tickers: Vec<String>,
}

#[test]
fn news_nests_lists_of_records() {
    assert_schema::<Article>(&[
        ("id", TEXT),
        ("title", TEXT),
        ("summary", TEXT),
        ("published_at", instant()),
        ("content_type", TEXT),
        ("provider.name", TEXT),
        ("provider.url", TEXT),
        ("url", TEXT),
        ("thumbnails.url", list(TEXT)),
        ("thumbnails.width", list(DataType::UInt32)),
        ("thumbnails.height", list(DataType::UInt32)),
        ("thumbnails.tag", list(TEXT)),
        ("tickers", list(TEXT)),
    ]);

    let rows: Vec<Article> = (0..300)
        .map(|index| Article {
            id: format!("{index:08x}"),
            title: format!("Headline {index}"),
            summary: (index % 3 == 0).then(|| "A longer summary of the story.".into()),
            published_at: at(index * 60),
            content_type: if index % 10 == 0 {
                ContentType::Video
            } else {
                ContentType::Story
            },
            provider: (index % 4 != 0).then(|| Provider {
                name: "Newswire".into(),
                url: (index % 8 != 1).then(|| "https://example.test".into()),
            }),
            url: Some(format!("https://example.test/{index}")),
            thumbnails: (0..index % 4)
                .map(|size| Thumbnail {
                    url: format!("https://example.test/{index}/{size}.jpg"),
                    width: 170 << size,
                    height: 128 << size,
                    tag: if size == 0 { "170x128" } else { "original" }.into(),
                })
                .collect(),
            tickers: (0..index % 3).map(|ticker| format!("T{ticker}")).collect(),
        })
        .collect();
    check(&rows);

    let df = rows.to_dataframe().unwrap();
    assert_eq!(
        strings(&df, "thumbnails.tag", 3),
        ["170x128", "original", "original"]
    );
    // A story without a provider has neither of the provider's columns.
    assert_eq!(df.column("provider.name").unwrap().null_count(), 75);
}

// --- info: a wide optional record, a list of records, and an open bag ------------

/// A value that is a number under one key and text under another.
#[derive(Clone, Debug)]
enum Scalar {
    Number(f64),
    Text(String),
    Flag(bool),
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(value) => write!(f, "{value}"),
            Self::Text(value) => f.write_str(value),
            Self::Flag(value) => write!(f, "{value}"),
        }
    }
}

#[derive(ToDataFrame, Clone)]
struct Officer {
    name: String,
    title: Option<String>,
    age: Option<u8>,
    total_pay: Option<Money>,
}

#[derive(ToDataFrame, Clone)]
struct Info {
    symbol: String,
    short_name: Option<String>,
    sector: Option<Sector>,
    market_cap: Option<Money>,
    trailing_pe: Option<f64>,
    employees: Option<u32>,
    officers: Vec<Officer>,
    /// Keys the typed fields do not cover, each rendered as text. The
    /// attribute applies to the map's values.
    #[df_derive(as_string)]
    extra: BTreeMap<String, Scalar>,
}

#[test]
fn info_is_a_wide_record_with_a_list_and_an_open_bag() {
    assert_schema::<Info>(&[
        ("symbol", TEXT),
        ("short_name", TEXT),
        ("sector", TEXT),
        ("market_cap.amount", DECIMAL),
        ("market_cap.currency", TEXT),
        ("trailing_pe", REAL),
        ("employees", DataType::UInt32),
        ("officers.name", list(TEXT)),
        ("officers.title", list(TEXT)),
        ("officers.age", list(DataType::UInt8)),
        ("officers.total_pay.amount", list(DECIMAL)),
        ("officers.total_pay.currency", list(TEXT)),
        ("extra.key", list(TEXT)),
        ("extra.value", list(TEXT)),
    ]);

    let rows: Vec<Info> = (0..200)
        .map(|index| Info {
            symbol: format!("S{index}"),
            short_name: (index % 6 != 0).then(|| format!("Company {index}")),
            sector: (index % 5 != 0).then_some(Sector::Technology),
            market_cap: (index % 3 != 0).then(|| money(index as i64 * 1_000_000)),
            trailing_pe: (index % 4 != 0).then_some(21.5),
            employees: (index % 7 != 0).then_some(1_000 + index as u32),
            officers: (0..index % 5)
                .map(|rank| Officer {
                    name: format!("Officer {rank}"),
                    title: (rank == 0).then(|| "Chief Executive Officer".into()),
                    age: (rank % 2 == 0).then_some(50 + rank as u8),
                    total_pay: (rank != 3).then(|| money(1_000_000 - rank as i64)),
                })
                .collect(),
            extra: match index % 3 {
                0 => BTreeMap::new(),
                1 => BTreeMap::from([("beta".to_owned(), Scalar::Number(1.25))]),
                _ => BTreeMap::from([
                    ("delisted".to_owned(), Scalar::Flag(false)),
                    ("last_trade".to_owned(), Scalar::Text("4:00PM EST".into())),
                ]),
            },
        })
        .collect();
    check(&rows);

    let df = rows.to_dataframe().unwrap();
    assert_eq!(strings(&df, "extra.key", 2), ["delisted", "last_trade"]);
    assert_eq!(strings(&df, "extra.value", 2), ["false", "4:00PM EST"]);
    assert_eq!(strings(&df, "extra.value", 1), ["1.25"]);
    assert_eq!(
        strings(&df, "officers.total_pay.currency", 4),
        ["USD", "EUR", "USD", "<null>"]
    );
}

// --- market summary: a map of records, and its entries as rows -------------------

#[derive(ToDataFrame, Clone)]
struct MarketQuote {
    price: Decimal,
    change_percent: Option<f64>,
    state: MarketState,
    #[df_derive(time_unit = "ns")]
    at: DateTime<Utc>,
}

#[derive(ToDataFrame, Clone)]
struct MarketSummary {
    region: String,
    quotes: BTreeMap<String, MarketQuote>,
}

/// One entry of the map as a row: the key beside the value's own columns.
#[derive(ToDataFrame)]
struct ExchangeQuote<'a> {
    exchange: &'a str,
    #[df_derive(flatten)]
    quote: &'a MarketQuote,
}

/// A batch keyed by symbol whose values are themselves keyed by field.
#[derive(ToDataFrame, Clone)]
struct Batch {
    requested: Vec<String>,
    fields: BTreeMap<String, BTreeMap<String, String>>,
}

fn summary(index: usize) -> MarketSummary {
    MarketSummary {
        region: format!("R{index}"),
        quotes: (0..index % 6)
            .map(|exchange| {
                (
                    format!("X{exchange}"),
                    MarketQuote {
                        price: Decimal::new(10_000 + (index * 10 + exchange) as i64, 2),
                        change_percent: (exchange % 2 == 0).then_some(0.5),
                        state: match exchange {
                            0 => MarketState::Regular,
                            1 => MarketState::Pre,
                            _ => MarketState::Closed,
                        },
                        at: at((index * 60 + exchange) as i64),
                    },
                )
            })
            .collect(),
    }
}

#[test]
fn a_market_summary_is_a_map_of_records_or_a_row_per_entry() {
    assert_schema::<MarketSummary>(&[
        ("region", TEXT),
        ("quotes.key", list(TEXT)),
        ("quotes.value.price", list(DECIMAL)),
        ("quotes.value.change_percent", list(REAL)),
        ("quotes.value.state", list(TEXT)),
        ("quotes.value.at", list(instant())),
    ]);
    assert_schema::<ExchangeQuote<'_>>(&[
        ("exchange", TEXT),
        ("price", DECIMAL),
        ("change_percent", REAL),
        ("state", TEXT),
        ("at", instant()),
    ]);
    assert_schema::<Batch>(&[
        ("requested", list(TEXT)),
        ("fields.key", list(TEXT)),
        ("fields.value.key", list(list(TEXT))),
        ("fields.value.value", list(list(TEXT))),
    ]);

    let summaries: Vec<MarketSummary> = (0..200).map(summary).collect();
    check(&summaries);

    // The entries of every map as one long table.
    let entries: Vec<ExchangeQuote<'_>> = summaries
        .iter()
        .flat_map(|summary| &summary.quotes)
        .map(|(exchange, quote)| ExchangeQuote { exchange, quote })
        .collect();
    check(&entries);
    let total: usize = summaries.iter().map(|summary| summary.quotes.len()).sum();
    assert_eq!(entries.to_dataframe().unwrap().shape(), (total, 5));

    let batches: Vec<Batch> = (0..150)
        .map(|index| Batch {
            requested: (0..index % 4).map(|symbol| format!("S{symbol}")).collect(),
            fields: (0..index % 4)
                .map(|symbol| {
                    let fields = (0..symbol + index % 2)
                        .map(|field| (format!("field{field}"), format!("{index}/{symbol}/{field}")))
                        .collect();
                    (format!("S{symbol}"), fields)
                })
                .collect(),
        })
        .collect();
    check(&batches);
}

// --- history: metadata with nested periods, and many tickers at once -------------

#[derive(ToDataFrame, Clone)]
struct Session {
    #[df_derive(time_unit = "ns")]
    start: DateTime<Utc>,
    #[df_derive(time_unit = "ns")]
    end: DateTime<Utc>,
}

#[derive(ToDataFrame, Clone)]
struct TradingPeriods {
    pre: Session,
    regular: Session,
    post: Session,
}

#[derive(ToDataFrame, Clone)]
struct HistoryMeta {
    symbol: String,
    exchange_time_zone: String,
    utc_offset_seconds: i32,
    current: TradingPeriods,
    valid_ranges: Vec<String>,
    /// A table embedded in the record: one row per past session.
    sessions: Vec<Session>,
}

#[derive(ToDataFrame, Clone)]
struct Candle {
    #[df_derive(time_unit = "ns")]
    ts: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: Option<u64>,
}

#[derive(ToDataFrame, Clone)]
struct TickerHistory {
    symbol: String,
    candles: Vec<Candle>,
}

/// One candle of one ticker. As a table it would have a group of columns per
/// ticker; here the ticker is a column, so any number of them fits one schema.
#[derive(ToDataFrame)]
struct TickerCandle<'a> {
    symbol: &'a str,
    #[df_derive(flatten)]
    candle: &'a Candle,
}

fn session(start: i64) -> Session {
    Session {
        start: at(start),
        end: at(start + 23_400),
    }
}

#[test]
fn history_metadata_nests_periods_and_embeds_a_table() {
    assert_schema::<HistoryMeta>(&[
        ("symbol", TEXT),
        ("exchange_time_zone", TEXT),
        ("utc_offset_seconds", DataType::Int32),
        ("current.pre.start", instant()),
        ("current.pre.end", instant()),
        ("current.regular.start", instant()),
        ("current.regular.end", instant()),
        ("current.post.start", instant()),
        ("current.post.end", instant()),
        ("valid_ranges", list(TEXT)),
        ("sessions.start", list(instant())),
        ("sessions.end", list(instant())),
    ]);

    let rows: Vec<HistoryMeta> = (0..120)
        .map(|index| HistoryMeta {
            symbol: format!("S{index}"),
            exchange_time_zone: "America/New_York".into(),
            utc_offset_seconds: -18_000,
            current: TradingPeriods {
                pre: session(index),
                regular: session(index + 19_800),
                post: session(index + 43_200),
            },
            valid_ranges: ["1d", "5d", "1mo", "max"][..index as usize % 5]
                .iter()
                .map(|&range| range.to_owned())
                .collect(),
            sessions: (0..index % 6).map(|past| session(past * 86_400)).collect(),
        })
        .collect();
    check(&rows);
}

#[test]
fn many_tickers_are_one_long_table_or_a_list_per_ticker() {
    assert_schema::<TickerHistory>(&[
        ("symbol", TEXT),
        ("candles.ts", list(instant())),
        ("candles.open", list(REAL)),
        ("candles.high", list(REAL)),
        ("candles.low", list(REAL)),
        ("candles.close", list(REAL)),
        ("candles.volume", list(DataType::UInt64)),
    ]);
    assert_schema::<TickerCandle<'_>>(&[
        ("symbol", TEXT),
        ("ts", instant()),
        ("open", REAL),
        ("high", REAL),
        ("low", REAL),
        ("close", REAL),
        ("volume", DataType::UInt64),
    ]);

    let histories: Vec<TickerHistory> = (0..40)
        .map(|ticker| TickerHistory {
            symbol: format!("T{ticker}"),
            // Tickers have histories of different lengths, some none at all.
            candles: (0..ticker * 13 % 250)
                .map(|bar| {
                    let price = 100.0 + f64::from(bar);
                    Candle {
                        ts: at(i64::from(bar) * 86_400),
                        open: price,
                        high: price + 1.0,
                        low: price - 1.0,
                        close: price + 0.5,
                        volume: (bar % 9 != 0).then_some(1_000 + bar as u64),
                    }
                })
                .collect(),
        })
        .collect();
    check(&histories);

    let long: Vec<TickerCandle<'_>> = histories
        .iter()
        .flat_map(|history| {
            history.candles.iter().map(|candle| TickerCandle {
                symbol: &history.symbol,
                candle,
            })
        })
        .collect();
    check(&long);

    // Both forms hold the same bars.
    let bars: usize = histories.iter().map(|history| history.candles.len()).sum();
    let df = long.to_dataframe().unwrap();
    assert_eq!(df.shape(), (bars, 7));
    let nested = histories.to_dataframe().unwrap();
    let exploded = nested
        .explode(
            [
                "candles.ts",
                "candles.open",
                "candles.high",
                "candles.low",
                "candles.close",
                "candles.volume",
            ],
            ExplodeOptions {
                empty_as_null: false,
                keep_nulls: false,
            },
        )
        .unwrap();
    assert_eq!(exploded.height(), bars);
    assert!(
        exploded.column("candles.close").unwrap().equals_missing(
            &df.column("close")
                .unwrap()
                .clone()
                .with_name("candles.close".into())
        )
    );
}
