// Features that let a type say once how it is stored, instead of annotating
// every field that holds it: `transparent` wrappers, `impl Columnar` for leaf
// types, `via` / `try_via` projections, custom builders through `with`, map
// fields, and errors that name the row and field they came from.

use crate::core::dataframe::builders::{PrimitiveBuilder, StrBuilder};
use crate::core::dataframe::{
    Bitmap, ColumnBuilder, Columnar, Fields, ToDataFrame, ToDataFrameVec,
};
use chrono::{DateTime, TimeZone, Utc};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::collections::{BTreeMap, HashMap};

fn names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .into_iter()
        .map(|name| name.as_str().to_owned())
        .collect()
}

// --- transparent wrappers ------------------------------------------------------

#[derive(ToDataFrame, Clone)]
#[df_derive(transparent)]
struct Px(f64);

#[derive(ToDataFrame, Clone)]
#[df_derive(transparent)]
struct Label {
    text: String,
}

#[derive(ToDataFrame, Clone)]
struct Quote {
    label: Label,
    price: Px,
    previous: Option<Px>,
    history: Vec<Px>,
}

#[test]
fn transparent_wrapper_takes_the_name_of_its_field() {
    let rows = vec![
        Quote {
            label: Label { text: "a".into() },
            price: Px(1.5),
            previous: None,
            history: vec![Px(1.0), Px(1.25)],
        },
        Quote {
            label: Label { text: "b".into() },
            price: Px(2.5),
            previous: Some(Px(2.0)),
            history: vec![],
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["label", "price", "previous", "history"]);
    assert_eq!(
        Quote::schema().unwrap(),
        vec![
            ("label".to_owned(), DataType::String),
            ("price".to_owned(), DataType::Float64),
            ("previous".to_owned(), DataType::Float64),
            (
                "history".to_owned(),
                DataType::List(Box::new(DataType::Float64))
            ),
        ]
    );
    assert_eq!(
        df.column("previous").unwrap().get(0).unwrap(),
        AnyValue::Null
    );
    assert_eq!(
        df.column("previous").unwrap().get(1).unwrap(),
        AnyValue::Float64(2.0)
    );
}

// --- a leaf type that declares its own encoding ----------------------------------

#[derive(Clone, Copy)]
enum Side {
    Buy,
    Sell,
}

impl AsRef<str> for Side {
    fn as_ref(&self) -> &str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
        }
    }
}

impl Columnar for Side {
    type Builder = StrBuilder;
}

#[derive(ToDataFrame)]
struct Order {
    side: Side,
    hedge: Option<Side>,
    legs: Vec<Side>,
}

#[test]
fn declared_leaf_needs_no_field_attribute() {
    let df = Order {
        side: Side::Buy,
        hedge: None,
        legs: vec![Side::Sell, Side::Buy],
    }
    .to_dataframe()
    .unwrap();
    assert_eq!(
        Order::schema().unwrap(),
        vec![
            ("side".to_owned(), DataType::String),
            ("hedge".to_owned(), DataType::String),
            (
                "legs".to_owned(),
                DataType::List(Box::new(DataType::String))
            ),
        ]
    );
    assert_eq!(
        df.column("side").unwrap().get(0).unwrap(),
        AnyValue::String("BUY")
    );
}

// --- projections ---------------------------------------------------------------

#[derive(ToDataFrame)]
#[df_derive(via = "InstrumentRow<'_>")]
struct Instrument {
    symbol: String,
    exchange: Option<String>,
}

#[derive(ToDataFrame)]
struct InstrumentRow<'a> {
    symbol: &'a str,
    exchange: Option<&'a str>,
    key: String,
}

impl<'a> From<&'a Instrument> for InstrumentRow<'a> {
    fn from(instrument: &'a Instrument) -> Self {
        Self {
            symbol: &instrument.symbol,
            exchange: instrument.exchange.as_deref(),
            key: format!(
                "{}@{}",
                instrument.symbol,
                instrument.exchange.as_deref().unwrap_or("-")
            ),
        }
    }
}

#[derive(ToDataFrame)]
struct Holding {
    instrument: Instrument,
    alternates: Vec<Instrument>,
    quantity: f64,
}

fn instrument(symbol: &str, exchange: Option<&str>) -> Instrument {
    Instrument {
        symbol: symbol.to_owned(),
        exchange: exchange.map(str::to_owned),
    }
}

#[test]
fn via_encodes_through_a_borrowed_row() {
    let rows = vec![instrument("AAPL", Some("XNAS")), instrument("BTC", None)];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["symbol", "exchange", "key"]);
    assert_eq!(
        df.column("key").unwrap().get(1).unwrap(),
        AnyValue::String("BTC@-")
    );
    assert_eq!(
        Instrument::schema().unwrap(),
        InstrumentRow::schema().unwrap()
    );

    let holding = Holding {
        instrument: instrument("MSFT", Some("XNAS")),
        alternates: vec![instrument("MSFT", Some("XETR"))],
        quantity: 3.0,
    };
    let df = holding.to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "instrument.symbol",
            "instrument.exchange",
            "instrument.key",
            "alternates.symbol",
            "alternates.exchange",
            "alternates.key",
            "quantity",
        ]
    );
}

#[derive(ToDataFrame)]
#[df_derive(via = "ActionRow")]
enum Action {
    Dividend { amount: f64 },
    Split { numerator: u32, denominator: u32 },
}

#[derive(ToDataFrame)]
struct ActionRow {
    kind: &'static str,
    amount: Option<f64>,
    numerator: Option<u32>,
    denominator: Option<u32>,
}

impl From<&Action> for ActionRow {
    fn from(action: &Action) -> Self {
        match *action {
            Action::Dividend { amount } => Self {
                kind: "dividend",
                amount: Some(amount),
                numerator: None,
                denominator: None,
            },
            Action::Split {
                numerator,
                denominator,
            } => Self {
                kind: "split",
                amount: None,
                numerator: Some(numerator),
                denominator: Some(denominator),
            },
        }
    }
}

#[test]
fn via_spreads_an_enum_over_columns() {
    let rows = [
        Action::Dividend { amount: 0.24 },
        Action::Split {
            numerator: 4,
            denominator: 1,
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["kind", "amount", "numerator", "denominator"]);
    assert_eq!(df.column("amount").unwrap().get(1).unwrap(), AnyValue::Null);
    assert_eq!(
        df.column("numerator").unwrap().get(1).unwrap(),
        AnyValue::UInt32(4)
    );
}

#[derive(ToDataFrame)]
#[df_derive(try_via = "CheckedRow<'_>")]
struct Checked {
    name: String,
    ratio: f64,
}

#[derive(ToDataFrame)]
struct CheckedRow<'a> {
    name: &'a str,
    ratio: f64,
}

impl<'a> TryFrom<&'a Checked> for CheckedRow<'a> {
    type Error = String;

    fn try_from(value: &'a Checked) -> Result<Self, Self::Error> {
        if value.ratio.is_finite() {
            Ok(Self {
                name: &value.name,
                ratio: value.ratio,
            })
        } else {
            Err(format!("ratio of `{}` is not finite", value.name))
        }
    }
}

#[test]
fn try_via_reports_the_failing_row() {
    let rows = vec![
        Checked {
            name: "ok".into(),
            ratio: 1.0,
        },
        Checked {
            name: "bad".into(),
            ratio: f64::NAN,
        },
    ];
    let error = rows.as_slice().to_dataframe().unwrap_err().to_string();
    assert!(
        error.contains("Checked[1]: ratio of `bad` is not finite"),
        "{error}"
    );
    assert_eq!(rows[..1].to_dataframe().unwrap().shape(), (1, 2));
}

// --- custom builders -------------------------------------------------------------

/// Stores a price as integer cents.
struct Cents(PrimitiveBuilder<i64>);

impl ColumnBuilder<f64> for Cents {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <PrimitiveBuilder<i64> as ColumnBuilder<i64>>::schema(name, out);
    }

    fn with_capacity(rows: usize) -> Self {
        Self(ColumnBuilder::<i64>::with_capacity(rows))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn push(&mut self, value: &f64) -> PolarsResult<()> {
        self.0.push(&((value * 100.0).round() as i64))
    }

    fn push_null(&mut self) {
        ColumnBuilder::<i64>::push_null(&mut self.0);
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

#[derive(ToDataFrame)]
struct Priced {
    #[df_derive(with = "Cents")]
    price: f64,
    #[df_derive(with = "Cents")]
    limit: Option<f64>,
}

#[test]
fn with_selects_a_custom_builder() {
    let rows = [
        Priced {
            price: 1.239,
            limit: None,
        },
        Priced {
            price: 2.0,
            limit: Some(2.5),
        },
    ];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.column("price").unwrap().dtype(), &DataType::Int64);
    assert_eq!(
        df.column("price").unwrap().get(0).unwrap(),
        AnyValue::Int64(124)
    );
    assert_eq!(df.column("limit").unwrap().get(0).unwrap(), AnyValue::Null);
    assert_eq!(
        df.column("limit").unwrap().get(1).unwrap(),
        AnyValue::Int64(250)
    );
}

// --- maps ----------------------------------------------------------------------

#[derive(ToDataFrame)]
struct Fund {
    name: String,
    weights: BTreeMap<String, f64>,
    ratings: Option<HashMap<String, u32>>,
}

#[test]
fn map_field_becomes_key_and_value_lists() {
    let fund = Fund {
        name: "f".into(),
        weights: BTreeMap::from([("tech".to_owned(), 0.6), ("energy".to_owned(), 0.4)]),
        ratings: None,
    };
    let df = fund.to_dataframe().unwrap();
    assert_eq!(
        Fund::schema().unwrap(),
        vec![
            ("name".to_owned(), DataType::String),
            (
                "weights.key".to_owned(),
                DataType::List(Box::new(DataType::String))
            ),
            (
                "weights.value".to_owned(),
                DataType::List(Box::new(DataType::Float64))
            ),
            (
                "ratings.key".to_owned(),
                DataType::List(Box::new(DataType::String))
            ),
            (
                "ratings.value".to_owned(),
                DataType::List(Box::new(DataType::UInt32))
            ),
        ]
    );
    let keys = df.column("weights.key").unwrap().get(0).unwrap();
    let AnyValue::List(keys) = keys else {
        panic!("expected a list, got {keys:?}");
    };
    let keys: Vec<_> = keys.str().unwrap().iter().flatten().collect();
    assert_eq!(keys, ["energy", "tech"]);
    assert_eq!(
        df.column("ratings.key").unwrap().get(0).unwrap(),
        AnyValue::Null
    );
}

// --- error context -------------------------------------------------------------

#[derive(ToDataFrame)]
struct Stamp {
    #[df_derive(time_unit = "ns")]
    at: DateTime<Utc>,
}

#[derive(ToDataFrame)]
struct Event {
    id: u32,
    stamp: Stamp,
    history: Vec<Stamp>,
}

#[test]
fn encoding_error_names_the_row_and_field() {
    let fine = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
    let too_far = Utc.with_ymd_and_hms(3000, 1, 1, 0, 0, 0).single().unwrap();
    let rows = vec![
        Event {
            id: 1,
            stamp: Stamp { at: fine },
            history: vec![],
        },
        Event {
            id: 2,
            stamp: Stamp { at: too_far },
            history: vec![],
        },
    ];
    let error = rows.as_slice().to_dataframe().unwrap_err().to_string();
    assert!(
        error.contains("Event[1].stamp.at: df-derive: DateTime<Tz> value is out of range"),
        "{error}"
    );

    let rows = vec![Event {
        id: 3,
        stamp: Stamp { at: fine },
        history: vec![Stamp { at: fine }, Stamp { at: too_far }],
    }];
    let error = rows.as_slice().to_dataframe().unwrap_err().to_string();
    assert!(error.contains("Event[0].history.at: "), "{error}");
}

// --- any iterator of rows --------------------------------------------------------

#[test]
fn rows_to_dataframe_accepts_any_iterator_of_references() {
    let rows = [Px(1.0), Px(2.0), Px(3.0)];
    let quotes: Vec<Quote> = rows
        .iter()
        .map(|price| Quote {
            label: Label { text: "x".into() },
            price: price.clone(),
            previous: None,
            history: vec![],
        })
        .collect();
    let df = Quote::rows_to_dataframe(quotes.iter().filter(|quote| quote.price.0 > 1.5)).unwrap();
    assert_eq!(df.height(), 2);
    assert_eq!(Quote::empty_dataframe().unwrap().shape(), (0, 4));
}
