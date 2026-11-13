//! Optional numbers whose empty slots follow no pattern.
//!
//! The other benchmarks leave a slot empty every so many rows, and a
//! processor's branch predictor learns a pattern like that. Real columns are
//! rarely so regular. Here whether a slot is empty comes from a hash of its
//! position, so an encoder that branches on it pays for every wrong guess.

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
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_irregular_nulls
}
criterion_main!(benches);
