//! Optional numbers, dates and times whose empty slots follow no pattern.
//!
//! The other benchmarks leave a slot empty every so many rows, and a
//! processor's branch predictor learns a pattern like that. Real columns are
//! rarely so regular. Here whether a slot is empty comes from a hash of its
//! position, so an encoder that branches on it pays for every wrong guess.

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Utc};
use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
use crate::bench_support::configure_criterion;
use df_derive::dataframe::ToDataFrameVec;

const N_ROWS: usize = 50_000;
const N_LISTS: usize = 16_000;

#[derive(ToDataFrame, Clone)]
struct Reading {
    id: u64,
    price: Option<f64>,
    bid: Option<f64>,
    ask: Option<f64>,
    volume: Option<u64>,
    count: Option<i32>,
    level: Option<u8>,
    ratio: Option<f32>,
    halted: Option<bool>,
}

#[derive(ToDataFrame, Clone)]
struct Samples {
    id: u64,
    counts: Vec<Option<i32>>,
    values: Vec<Option<f64>>,
}

#[derive(ToDataFrame, Clone)]
struct Appointment {
    id: u64,
    day: Option<NaiveDate>,
    start: Option<NaiveTime>,
    booked: Option<DateTime<Utc>>,
    local: Option<NaiveDateTime>,
    length: Option<TimeDelta>,
    wait: Option<std::time::Duration>,
}

#[derive(ToDataFrame, Clone)]
struct Calendar {
    id: u64,
    days: Vec<Option<NaiveDate>>,
    stamps: Vec<Option<DateTime<Utc>>>,
}

/// A number that looks random, from a row and a place in it.
const fn mixed(row: usize, place: usize) -> u64 {
    let mut x = (row as u64).wrapping_mul(11).wrapping_add(place as u64 + 1);
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x >> 40
}

/// The value for a slot, or nothing for about one slot in six.
fn slot<T>(row: usize, place: usize, value: T) -> Option<T> {
    (!mixed(row, place).is_multiple_of(6)).then_some(value)
}

fn generate_readings() -> Vec<Reading> {
    (0..N_ROWS)
        .map(|row| {
            let n = u32::try_from(row).unwrap();
            let price = f64::from(n).mul_add(0.01, 100.0);
            Reading {
                id: row as u64,
                price: slot(row, 0, price),
                bid: slot(row, 1, price - 0.5),
                ask: slot(row, 2, price + 0.5),
                volume: slot(row, 3, 1_000 + row as u64),
                count: slot(row, 4, i32::try_from(row % 500).unwrap() - 250),
                level: slot(row, 5, u8::try_from(row % 200).unwrap()),
                ratio: slot(row, 6, f32::from(u16::try_from(row % 1_000).unwrap()) / 7.0),
                halted: slot(row, 7, row % 2 == 0),
            }
        })
        .collect()
}

fn generate_samples() -> Vec<Samples> {
    (0..N_LISTS)
        .map(|row| {
            // Two to five elements, in no order either.
            let len = 2 + usize::try_from(mixed(row, 99) % 4).unwrap();
            let n = i32::try_from(row).unwrap();
            Samples {
                id: row as u64,
                counts: (0..len)
                    .map(|item| slot(row, item, n + i32::try_from(item).unwrap()))
                    .collect(),
                values: (0..len)
                    .map(|item| slot(row, item + 8, f64::from(n) * 0.5))
                    .collect(),
            }
        })
        .collect()
}

fn day(row: usize) -> NaiveDate {
    NaiveDate::from_num_days_from_ce_opt(730_000 + i32::try_from(row % 5_000).unwrap()).unwrap()
}

fn instant(row: usize) -> DateTime<Utc> {
    DateTime::from_timestamp(1_600_000_000 + i64::try_from(row).unwrap(), 0).unwrap()
}

fn generate_appointments() -> Vec<Appointment> {
    (0..N_ROWS)
        .map(|row| {
            let seconds = u32::try_from(row % 86_400).unwrap();
            Appointment {
                id: row as u64,
                day: slot(row, 0, day(row)),
                start: slot(
                    row,
                    1,
                    NaiveTime::from_num_seconds_from_midnight_opt(seconds, 0).unwrap(),
                ),
                booked: slot(row, 2, instant(row)),
                local: slot(row, 3, instant(row).naive_utc()),
                length: slot(row, 4, TimeDelta::seconds(i64::from(seconds))),
                wait: slot(row, 5, std::time::Duration::from_millis(row as u64)),
            }
        })
        .collect()
}

fn generate_calendars() -> Vec<Calendar> {
    (0..N_LISTS)
        .map(|row| {
            let len = 2 + usize::try_from(mixed(row, 99) % 4).unwrap();
            Calendar {
                id: row as u64,
                days: (0..len)
                    .map(|item| slot(row, item, day(row + item)))
                    .collect(),
                stamps: (0..len)
                    .map(|item| slot(row, item + 8, instant(row + item)))
                    .collect(),
            }
        })
        .collect()
}

fn benchmark_irregular_nulls(c: &mut Criterion) {
    let readings = generate_readings();
    c.bench_function("irregular_nulls_numbers_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&readings).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let samples = generate_samples();
    c.bench_function("irregular_nulls_number_lists_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&samples).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let appointments = generate_appointments();
    c.bench_function("irregular_nulls_dates_and_times_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&appointments).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let calendars = generate_calendars();
    c.bench_function("irregular_nulls_date_lists_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&calendars).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_irregular_nulls
}
criterion_main!(benches);
