//! Polars keeps its global set of categories for as long as a categorical
//! column is alive, and starts another with the first column after that. A
//! text has one code in one set and may have another in the next.
//!
//! An enum stored as categories has the codes of its names kept from one
//! column to the next, for the columns that are selected from a column of
//! the names: those of more rows than the enum has names. The test here
//! states that the kept codes are those of the set in use. It is alone in its
//! file, and so in its process: a column that another test held at the wrong
//! moment would keep the first set alive.

#![cfg(feature = "categorical")]

use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrameVec;
use polars::prelude::*;

#[derive(ToDataFrame, Clone, Copy)]
#[df_derive(categorical)]
enum Status {
    Open,
    Filled,
}

#[derive(ToDataFrame)]
struct Order {
    status: Status,
    earlier: Option<Status>,
}

#[derive(ToDataFrame)]
struct Note {
    #[df_derive(categorical)]
    text: String,
}

/// Five orders: more rows than `Status` has names.
fn orders() -> DataFrame {
    [
        (Status::Filled, Some(Status::Open)),
        (Status::Open, None),
        (Status::Open, Some(Status::Filled)),
        (Status::Filled, None),
        (Status::Filled, Some(Status::Filled)),
    ]
    .map(|(status, earlier)| Order { status, earlier })
    .as_slice()
    .to_dataframe()
    .unwrap()
}

fn strings(df: &DataFrame, name: &str) -> Vec<Option<String>> {
    df.column(name)
        .unwrap()
        .cast(&DataType::String)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|text| text.map(str::to_owned))
        .collect()
}

fn codes(df: &DataFrame, name: &str) -> Vec<Option<u32>> {
    df.column(name)
        .unwrap()
        .cat32()
        .unwrap()
        .physical()
        .iter()
        .collect()
}

fn assert_orders(df: &DataFrame) {
    let text = |names: [Option<&str>; 5]| names.map(|name| name.map(str::to_owned));
    assert_eq!(
        strings(df, "status"),
        text([
            Some("Filled"),
            Some("Open"),
            Some("Open"),
            Some("Filled"),
            Some("Filled"),
        ])
    );
    assert_eq!(
        strings(df, "earlier"),
        text([Some("Open"), None, Some("Filled"), None, Some("Filled")])
    );
}

#[test]
fn the_codes_of_a_name_are_those_of_the_set_in_use() {
    let first = orders();
    assert_orders(&first);
    let in_first_set = codes(&first, "status");

    // While a column is alive the set is, and a name keeps its code.
    let beside = orders();
    assert_orders(&beside);
    assert_eq!(codes(&beside, "status"), in_first_set);

    drop(first);
    drop(beside);

    // No categorical column is left, so the next one starts a new set. Other
    // text is stored first and takes the codes the names had.
    let notes = ["zero", "one", "two", "Open"]
        .map(|text| Note {
            text: text.to_owned(),
        })
        .as_slice()
        .to_dataframe()
        .unwrap();
    let second = orders();
    assert_orders(&second);
    assert_eq!(
        strings(&notes, "text"),
        ["zero", "one", "two", "Open"].map(|text| Some(text.to_owned()))
    );

    // The names have other codes now, which is what makes this a test: had
    // the set been kept, the kept codes would be right without being looked
    // up again.
    assert_ne!(codes(&second, "status"), in_first_set);
    // A name and the same text stored as text are one category.
    assert_eq!(codes(&second, "status")[1], codes(&notes, "text")[3]);

    // And once more, with the names stored first.
    drop(second);
    drop(notes);
    let third = orders();
    assert_orders(&third);
}
