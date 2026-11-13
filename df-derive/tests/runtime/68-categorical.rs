// A categorical column holds a code for each row and each distinct string
// once. These tests state what `categorical` stores, how it goes with the
// two ways of reading text, and that every such column shares one set of
// categories.

use std::fmt;

use df_derive::ToDataFrame;
use df_derive::dataframe::builders::CategoricalBuilder;
use df_derive::dataframe::{Columnar, ToDataFrame, ToDataFrameVec};
use polars::prelude::*;

#[derive(Clone, Copy)]
enum Venue {
    Lit,
    Dark,
}

impl AsRef<str> for Venue {
    fn as_ref(&self) -> &str {
        match self {
            Self::Lit => "LIT",
            Self::Dark => "DARK",
        }
    }
}

#[derive(Clone, Copy)]
enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(match self {
            Self::Buy => "buy",
            Self::Sell => "sell",
        })
    }
}

/// A type that says once, for every field that holds it, that it is stored
/// as a category.
#[derive(Clone, Copy)]
struct Currency(&'static str);

impl AsRef<str> for Currency {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl Columnar for Currency {
    type Builder = CategoricalBuilder;
}

#[derive(ToDataFrame)]
struct Fill {
    #[df_derive(categorical)]
    venue: Venue,
    #[df_derive(as_string, categorical)]
    side: Side,
    #[df_derive(as_str, categorical)]
    symbol: String,
    #[df_derive(categorical)]
    note: Option<String>,
    #[df_derive(categorical)]
    tags: Vec<Option<&'static str>>,
    currency: Currency,
    maybe_currency: Option<Currency>,
    plain: String,
}

fn fills() -> Vec<Fill> {
    vec![
        Fill {
            venue: Venue::Lit,
            side: Side::Buy,
            symbol: "AAPL".into(),
            note: Some("opening".into()),
            tags: vec![Some("fast"), None, Some("LIT")],
            currency: Currency("USD"),
            maybe_currency: Some(Currency("EUR")),
            plain: "a".into(),
        },
        Fill {
            venue: Venue::Dark,
            side: Side::Sell,
            symbol: "MSFT".into(),
            note: None,
            tags: vec![],
            currency: Currency("EUR"),
            maybe_currency: None,
            plain: "b".into(),
        },
        Fill {
            venue: Venue::Lit,
            side: Side::Sell,
            symbol: "AAPL".into(),
            note: Some("opening".into()),
            tags: vec![Some("fast")],
            currency: Currency("USD"),
            maybe_currency: Some(Currency("USD")),
            plain: "a".into(),
        },
    ]
}

fn is_categorical(dtype: &DataType) -> bool {
    matches!(dtype, DataType::Categorical(_, _))
}

/// The strings of a categorical column, one per row.
fn strings(column: &Column) -> Vec<Option<String>> {
    column
        .cat32()
        .unwrap()
        .iter_str()
        .map(|text| text.map(str::to_owned))
        .collect()
}

/// The codes of a categorical column, one per row.
fn codes(column: &Column) -> Vec<Option<u32>> {
    column.cat32().unwrap().physical().iter().collect()
}

fn text(values: &[Option<&str>]) -> Vec<Option<String>> {
    values
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect()
}

#[test]
fn a_categorical_field_is_a_categorical_column() {
    let df = fills().as_slice().to_dataframe().unwrap();

    for name in [
        "venue",
        "side",
        "symbol",
        "note",
        "currency",
        "maybe_currency",
    ] {
        assert!(
            is_categorical(df.column(name).unwrap().dtype()),
            "column {name} is {:?}",
            df.column(name).unwrap().dtype()
        );
    }
    let DataType::List(inner) = df.column("tags").unwrap().dtype() else {
        panic!("tags is not a list");
    };
    assert!(is_categorical(inner));
    // A field without the attribute is the string column it always was.
    assert_eq!(df.column("plain").unwrap().dtype(), &DataType::String);

    // The schema names the same types without building a frame.
    let schema = Fill::schema().unwrap();
    for (name, dtype) in &schema {
        assert_eq!(dtype, df.column(name).unwrap().dtype(), "column {name}");
    }
}

#[test]
fn the_column_reads_back_as_the_text_it_was_given() {
    let df = fills().as_slice().to_dataframe().unwrap();

    assert_eq!(
        strings(df.column("venue").unwrap()),
        text(&[Some("LIT"), Some("DARK"), Some("LIT")])
    );
    // With `as_string` the text is what `Display` writes.
    assert_eq!(
        strings(df.column("side").unwrap()),
        text(&[Some("buy"), Some("sell"), Some("sell")])
    );
    assert_eq!(
        strings(df.column("symbol").unwrap()),
        text(&[Some("AAPL"), Some("MSFT"), Some("AAPL")])
    );
    assert_eq!(
        strings(df.column("currency").unwrap()),
        text(&[Some("USD"), Some("EUR"), Some("USD")])
    );
}

#[test]
fn an_empty_slot_is_a_null() {
    let df = fills().as_slice().to_dataframe().unwrap();

    assert_eq!(
        strings(df.column("note").unwrap()),
        text(&[Some("opening"), None, Some("opening")])
    );
    assert_eq!(
        strings(df.column("maybe_currency").unwrap()),
        text(&[Some("EUR"), None, Some("USD")])
    );

    // The lists hold three tags, none and one: four values in all, kept
    // together in one categorical column behind the lists.
    let tags = df.column("tags").unwrap().list().unwrap();
    let lens: Vec<usize> = (0..3)
        .map(|row| tags.get_as_series(row).unwrap().len())
        .collect();
    assert_eq!(lens, [3, 0, 1]);
    let values: Column = tags.get_inner().into();
    assert_eq!(
        strings(&values),
        text(&[Some("fast"), None, Some("LIT"), Some("fast")])
    );
}

#[test]
fn equal_text_has_one_code_in_every_column() {
    let df = fills().as_slice().to_dataframe().unwrap();

    // Within a column, a string that repeats repeats its code.
    let venue = codes(df.column("venue").unwrap());
    assert_eq!(venue[0], venue[2]);
    assert_ne!(venue[0], venue[1]);

    // Across columns too: every categorical column shares one set of
    // categories, so "USD" is the same code wherever it is stored, and so is
    // "LIT", which one row holds as a venue and as a tag.
    let currency = codes(df.column("currency").unwrap());
    let maybe = codes(df.column("maybe_currency").unwrap());
    assert_eq!(currency[0], maybe[2]);
    assert_eq!(currency[1], maybe[0]);

    let tags = df.column("tags").unwrap().list().unwrap();
    let values: Column = tags.get_inner().into();
    assert_eq!(codes(&values)[2], venue[0]);

    // And across frames: a second frame's columns have the same type, and
    // the two stack without a cast.
    let again = fills().as_slice().to_dataframe().unwrap();
    assert_eq!(
        df.column("venue").unwrap().dtype(),
        again.column("venue").unwrap().dtype()
    );
    let stacked = df.vstack(&again).unwrap();
    assert_eq!(stacked.height(), 6);
    assert_eq!(codes(stacked.column("venue").unwrap())[3], venue[0]);
}

#[test]
fn a_frame_of_no_rows_keeps_the_type() {
    let empty = Fill::empty_dataframe().unwrap();
    assert_eq!(empty.height(), 0);
    assert!(is_categorical(empty.column("venue").unwrap().dtype()));

    let none: Vec<Fill> = Vec::new();
    let df = none.as_slice().to_dataframe().unwrap();
    assert!(is_categorical(df.column("side").unwrap().dtype()));
}

struct Broken;

impl fmt::Display for Broken {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}

#[derive(ToDataFrame)]
struct Report {
    #[df_derive(as_string, categorical)]
    status: Broken,
}

#[test]
fn a_value_that_cannot_be_formatted_is_an_error() {
    let error = [Report { status: Broken }]
        .as_slice()
        .to_dataframe()
        .unwrap_err()
        .to_string();
    assert!(error.contains("Report[0].status"), "{error}");
    assert!(error.contains("formatting failed"), "{error}");
}
