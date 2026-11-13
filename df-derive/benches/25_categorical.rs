//! Text that repeats, stored as categories and as strings.
//!
//! A categorical column looks every value up in the set of categories, where
//! a string column only appends it. The two targets here encode the same rows
//! both ways, so that what the lookup costs is on record.

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
use crate::bench_support::configure_criterion;
use df_derive::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;

const CURRENCIES: [&str; 20] = [
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "PLN", "CZK",
    "HUF", "TRY", "ZAR", "MXN", "BRL", "SGD", "HKD",
];

#[derive(ToDataFrame, Clone)]
struct AsCategories {
    id: u64,
    #[df_derive(categorical)]
    currency: &'static str,
    #[df_derive(categorical)]
    venue: String,
    #[df_derive(categorical)]
    side: &'static str,
    #[df_derive(categorical)]
    note: Option<&'static str>,
}

#[derive(ToDataFrame, Clone)]
struct AsStrings {
    id: u64,
    currency: &'static str,
    venue: String,
    side: &'static str,
    note: Option<&'static str>,
}

/// A number that looks random, from a row and a place in it.
const fn mixed(row: usize, place: usize) -> usize {
    let mut x = (row as u64).wrapping_mul(11).wrapping_add(place as u64 + 1);
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (x >> 40) as usize
}

/// The text of one row: a currency out of twenty, a venue out of a hundred,
/// one of two sides, and a note that is missing from one row in six.
fn text(row: usize) -> (&'static str, String, &'static str, Option<&'static str>) {
    (
        CURRENCIES[mixed(row, 0) % CURRENCIES.len()],
        format!("VENUE{:02}", mixed(row, 1) % 100),
        if mixed(row, 2).is_multiple_of(2) {
            "BUY"
        } else {
            "SELL"
        },
        (!mixed(row, 3).is_multiple_of(6)).then_some("regular way"),
    )
}

fn benchmark_categorical(c: &mut Criterion) {
    let categories: Vec<AsCategories> = (0..N_ROWS)
        .map(|row| {
            let (currency, venue, side, note) = text(row);
            AsCategories {
                id: row as u64,
                currency,
                venue,
                side,
                note,
            }
        })
        .collect();
    c.bench_function("text_as_categories_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&categories).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let strings: Vec<AsStrings> = (0..N_ROWS)
        .map(|row| {
            let (currency, venue, side, note) = text(row);
            AsStrings {
                id: row as u64,
                currency,
                venue,
                side,
                note,
            }
        })
        .collect();
    c.bench_function("text_as_strings_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&strings).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_categorical
}
criterion_main!(benches);
