//! Enums whose variants hold no data, stored three ways.
//!
//! Such an enum is an `Enum` column by default: a row holds the position of
//! its variant, and nothing is looked up. A field can ask for the variant's
//! name instead, as a string or as a category. The first three targets encode
//! the same rows each way, so that what each costs is on record.
//!
//! The last two encode one row. An `Enum` column is selected from a column
//! of the type's variants, which is made once for a type and kept. A frame of
//! one row is where making it for every column would show, and where what a
//! selection costs to set up does.

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
use crate::bench_support::configure_criterion;
use df_derive::dataframe::{ToDataFrame as _, ToDataFrameVec};

const N_ROWS: usize = 100_000;

/// An enum that is a column of its own, and whose variants are also text.
macro_rules! variants {
    ($name:ident { $($variant:ident),* $(,)? }) => {
        #[derive(ToDataFrame, Clone, Copy)]
        enum $name {
            $($variant),*
        }

        impl $name {
            const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// The variant a number picks.
            const fn pick(number: usize) -> Self {
                Self::ALL[number % Self::ALL.len()]
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                match self {
                    $(Self::$variant => stringify!($variant)),*
                }
            }
        }
    };
}

variants!(Currency {
    Usd,
    Eur,
    Gbp,
    Jpy,
    Chf,
    Cad,
    Aud,
    Nzd,
    Sek,
    Nok,
    Dkk,
    Pln,
    Czk,
    Huf,
    Try,
    Zar,
    Mxn,
    Brl,
    Sgd,
    Hkd,
});
variants!(Side { Buy, Sell });
variants!(Kind {
    Market,
    Limit,
    Stop,
    StopLimit,
    Iceberg,
    Pegged,
});
variants!(State {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
});

#[derive(ToDataFrame, Clone)]
struct AsEnums {
    id: u64,
    currency: Currency,
    side: Side,
    kind: Kind,
    state: Option<State>,
}

#[derive(ToDataFrame, Clone)]
struct AsStrings {
    id: u64,
    #[df_derive(as_str)]
    currency: Currency,
    #[df_derive(as_str)]
    side: Side,
    #[df_derive(as_str)]
    kind: Kind,
    #[df_derive(as_str)]
    state: Option<State>,
}

#[derive(ToDataFrame, Clone)]
struct AsCategories {
    id: u64,
    #[df_derive(categorical)]
    currency: Currency,
    #[df_derive(categorical)]
    side: Side,
    #[df_derive(categorical)]
    kind: Kind,
    #[df_derive(categorical)]
    state: Option<State>,
}

/// A number that looks random, from a row and a place in it.
const fn mixed(row: usize, place: usize) -> usize {
    let mut x = (row as u64).wrapping_mul(11).wrapping_add(place as u64 + 1);
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (x >> 40) as usize
}

/// The variants of one row. The state is missing from one row in six.
fn variants(row: usize) -> (Currency, Side, Kind, Option<State>) {
    (
        Currency::pick(mixed(row, 0)),
        Side::pick(mixed(row, 1)),
        Kind::pick(mixed(row, 2)),
        (!mixed(row, 3).is_multiple_of(6)).then(|| State::pick(mixed(row, 4))),
    )
}

fn as_enums(row: usize) -> AsEnums {
    let (currency, side, kind, state) = variants(row);
    AsEnums {
        id: row as u64,
        currency,
        side,
        kind,
        state,
    }
}

fn as_strings(row: usize) -> AsStrings {
    let (currency, side, kind, state) = variants(row);
    AsStrings {
        id: row as u64,
        currency,
        side,
        kind,
        state,
    }
}

fn as_categories(row: usize) -> AsCategories {
    let (currency, side, kind, state) = variants(row);
    AsCategories {
        id: row as u64,
        currency,
        side,
        kind,
        state,
    }
}

fn benchmark_enum_columns(c: &mut Criterion) {
    let enums: Vec<AsEnums> = (0..N_ROWS).map(as_enums).collect();
    c.bench_function("variants_as_enums_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&enums).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let strings: Vec<AsStrings> = (0..N_ROWS).map(as_strings).collect();
    c.bench_function("variants_as_strings_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&strings).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let categories: Vec<AsCategories> = (0..N_ROWS).map(as_categories).collect();
    c.bench_function("variants_as_categories_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&categories).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let one = as_enums(7);
    c.bench_function("one_row_of_enums_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&one).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let one = as_strings(7);
    c.bench_function("one_row_of_strings_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&one).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_enum_columns
}
criterion_main!(benches);
