//! The fast path has to agree with the definition.
//!
//! Every builder defines its encoding one slot at a time, in `push`. Rows are
//! actually encoded a block at a time, through `measure`, `reserve` and
//! `extend`, each field by a loop of its own. These tests generate rows that
//! reach every builder, encode them both ways, and require equal frames, at
//! sizes that land on and around block boundaries. Errors get the same
//! treatment: wherever the bad value sits, the error names the first row
//! that pushing one at a time would have rejected.

// The casts are how the generator turns its integers into every other type,
// and it has no use for being `const`. The doubled wrappers are the shapes
// under test.
#![allow(
    clippy::box_collection,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    clippy::option_option,
    clippy::too_many_lines
)]

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, TimeZone, Utc};
use df_derive::ToDataFrame;
use df_derive::dataframe::builders::PrimitiveBuilder;
use df_derive::dataframe::{Bitmap, ColumnBuilder, Fields, ToDataFrame, ToDataFrameVec};
use polars::prelude::*;
use rust_decimal::Decimal;

#[path = "support/agreement.rs"]
mod agreement;
use agreement::{assert_same, check, first_rejected, one_at_a_time};

// --- generated data ------------------------------------------------------------

/// A small deterministic generator (`SplitMix64`), so a failure reproduces.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    /// Integers of every magnitude, both signs.
    fn int(&mut self) -> i64 {
        (self.next() as i64) >> self.below(64)
    }

    fn float(&mut self) -> f64 {
        self.int() as f64 / 128.0
    }

    /// Empty, short enough to be stored inline in a string view, or longer.
    fn text(&mut self) -> String {
        let len = match self.below(10) {
            0 => 0,
            1..=6 => 1 + self.below(11),
            _ => 13 + self.below(40),
        };
        (0..len)
            .map(|_| char::from(b'a' + self.below(26) as u8))
            .collect()
    }

    /// Mostly short, often empty, now and then long enough to span several
    /// blocks of elements.
    fn len(&mut self) -> usize {
        let len = match self.below(100) {
            0..20 => 0,
            20..90 => 1 + self.below(4),
            90..99 => 5 + self.below(40),
            _ => 300 + self.below(900),
        };
        len as usize
    }

    fn option<T>(&mut self, make: impl FnOnce(&mut Self) -> T) -> Option<T> {
        self.chance(70).then(|| make(self))
    }

    fn list<T>(&mut self, make: impl FnMut(&mut Self) -> T) -> Vec<T> {
        let len = self.len();
        self.list_of(len, make)
    }

    fn list_of<T>(&mut self, len: usize, mut make: impl FnMut(&mut Self) -> T) -> Vec<T> {
        (0..len).map(|_| make(self)).collect()
    }

    fn word(&mut self) -> &'static str {
        const WORDS: [&str; 5] = ["", "bid", "ask", "a considerably longer word", "mid"];
        WORDS[self.below(WORDS.len() as u64) as usize]
    }

    /// Any instant a nanosecond timestamp can hold.
    fn instant(&mut self) -> DateTime<Utc> {
        Utc.timestamp_nanos(self.int())
    }

    fn date(&mut self) -> NaiveDate {
        NaiveDate::from_num_days_from_ce_opt(700_000 + self.below(40_000) as i32).unwrap()
    }

    fn time(&mut self) -> NaiveTime {
        let seconds = self.below(86_400) as u32;
        let nanos = self.below(1_000_000_000) as u32;
        NaiveTime::from_num_seconds_from_midnight_opt(seconds, nanos).unwrap()
    }

    /// Up to thirteen digits and five decimals: fits every decimal column
    /// declared below, and scales both up and down on the way in.
    fn decimal(&mut self) -> Decimal {
        Decimal::new(self.int() >> 20, self.below(6) as u32)
    }

    fn side(&mut self) -> Side {
        if self.chance(50) {
            Side::Buy
        } else {
            Side::Sell
        }
    }

    #[cfg(feature = "categorical")]
    fn phase(&mut self) -> Phase {
        match self.below(3) {
            0 => Phase::Open,
            1 => Phase::Halted,
            _ => Phase::Closed,
        }
    }

    fn leaf(&mut self) -> Leaf {
        Leaf {
            id: self.next() as u32,
            weight: self.float(),
            tag: self.option(Self::text),
            hits: self.list(|g| g.next() as u16),
        }
    }

    fn event(&mut self) -> Event {
        match self.below(3) {
            0 => Event::Trade {
                price: self.float(),
                venue: self.text(),
            },
            1 => Event::Halt {
                reasons: self.list(Self::text),
            },
            _ => Event::Tick,
        }
    }
}

// --- row types -----------------------------------------------------------------

#[derive(ToDataFrame, Clone)]
struct Leaf {
    id: u32,
    weight: f64,
    tag: Option<String>,
    hits: Vec<u16>,
}

#[derive(ToDataFrame, Clone)]
struct Place {
    lat: f32,
    lon: f32,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(transparent)]
struct Meters(f64);

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

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Buy => "buy",
            Self::Sell => "sell",
        })
    }
}

/// An enum that is one column: its variants hold no data.
#[cfg(feature = "categorical")]
#[derive(ToDataFrame, Clone, Copy)]
enum Phase {
    Open,
    #[df_derive(rename = "halted")]
    Halted,
    Closed,
}

/// An enum spread over columns through a row that borrows from it.
#[derive(ToDataFrame, Clone)]
#[df_derive(via = "EventRow<'_>")]
enum Event {
    Trade { price: f64, venue: String },
    Halt { reasons: Vec<String> },
    Tick,
}

#[derive(ToDataFrame)]
struct EventRow<'a> {
    kind: &'static str,
    price: Option<f64>,
    venue: Option<&'a str>,
    reasons: &'a [String],
}

impl<'a> From<&'a Event> for EventRow<'a> {
    fn from(event: &'a Event) -> Self {
        let (kind, price, venue, reasons) = match event {
            Event::Trade { price, venue } => ("trade", Some(*price), Some(venue.as_str()), None),
            Event::Halt { reasons } => ("halt", None, None, Some(reasons.as_slice())),
            Event::Tick => ("tick", None, None, None),
        };
        Self {
            kind,
            price,
            venue,
            reasons: reasons.unwrap_or_default(),
        }
    }
}

/// A builder that implements only what it must, so the provided `measure`
/// and `extend` are compared with `push` too.
struct Cents(PrimitiveBuilder<i64>);

impl ColumnBuilder<f64> for Cents {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <PrimitiveBuilder<i64> as ColumnBuilder<i64>>::schema(name, out);
    }

    fn new() -> Self {
        Self(ColumnBuilder::<i64>::new())
    }

    fn reserve(&mut self, additional: usize) {
        ColumnBuilder::<i64>::reserve(&mut self.0, additional);
    }

    fn push(&mut self, value: Option<&f64>) -> PolarsResult<()> {
        let cents = value.map(|value| (value * 100.0).round() as i64);
        self.0.push(cents.as_ref())
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

/// One field for every builder and every way of nesting them.
#[derive(ToDataFrame)]
struct Everything<'a> {
    // Native values.
    a_i8: i8,
    a_i16: i16,
    a_i32: i32,
    a_i64: i64,
    a_i128: i128,
    a_u8: u8,
    a_u16: u16,
    a_u32: u32,
    a_u64: u64,
    a_u128: u128,
    a_f32: f32,
    a_f64: f64,
    a_bool: bool,
    a_usize: usize,
    a_isize: isize,
    non_zero: NonZeroU32,
    unit: (),
    #[df_derive(skip)]
    #[allow(dead_code)]
    skipped: u8,

    // Text.
    text: String,
    borrowed: &'a str,
    cow: Cow<'a, str>,
    #[df_derive(as_str)]
    side: Side,
    #[df_derive(as_string)]
    shown: Side,
    #[df_derive(as_binary)]
    bytes: Vec<u8>,

    // Text stored as categories.
    #[cfg(feature = "categorical")]
    #[df_derive(categorical)]
    cat_side: Side,
    #[cfg(feature = "categorical")]
    #[df_derive(as_string, categorical)]
    cat_shown: Side,
    #[cfg(feature = "categorical")]
    #[df_derive(categorical)]
    o_cat: Option<&'a str>,
    #[cfg(feature = "categorical")]
    #[df_derive(categorical)]
    l_o_cat: Vec<Option<String>>,

    // Variants stored by their place among the names.
    #[cfg(feature = "categorical")]
    phase: Phase,
    #[cfg(feature = "categorical")]
    o_phase: Option<Phase>,
    #[cfg(feature = "categorical")]
    l_o_phase: Vec<Option<Phase>>,

    // Options.
    o_i64: Option<i64>,
    o_bool: Option<bool>,
    o_text: Option<String>,
    o_o_i32: Option<Option<i32>>,
    #[df_derive(as_str)]
    o_side: Option<Side>,
    #[df_derive(as_binary)]
    o_bytes: Option<Vec<u8>>,

    // Lists.
    l_i32: Vec<i32>,
    l_bool: Vec<bool>,
    l_text: Vec<String>,
    l_o_i32: Vec<Option<i32>>,
    l_o_bool: Vec<Option<bool>>,
    l_o_text: Vec<Option<String>>,
    l_l_i32: Vec<Vec<i32>>,
    l_o_l_u8: Vec<Option<Vec<u8>>>,
    o_l_o_f64: Option<Vec<Option<f64>>>,
    #[df_derive(as_string)]
    l_shown: Vec<Side>,

    // Pointers.
    boxed: Box<i64>,
    o_boxed: Option<Box<String>>,
    rc: Rc<f32>,
    arc_list: Arc<Vec<u32>>,
    boxed_slice: Box<[i16]>,

    // Structs.
    leaf: Leaf,
    o_leaf: Option<Leaf>,
    b_leaf: Box<Leaf>,
    l_leaf: Vec<Leaf>,
    l_o_leaf: Vec<Option<Leaf>>,
    o_l_o_leaf: Option<Vec<Option<Leaf>>>,
    l_l_leaf: Vec<Vec<Leaf>>,
    #[df_derive(flatten)]
    place: Place,
    #[df_derive(flatten(prefix = "home"))]
    home: Option<Place>,

    // Tuples and maps.
    pair: (i32, String),
    l_pair: Vec<(u8, Option<f64>)>,
    map: BTreeMap<String, f64>,
    map_of_lists: BTreeMap<u32, Vec<i64>>,
    o_map_of_leaves: Option<BTreeMap<String, Leaf>>,

    // Time.
    at: DateTime<Utc>,
    #[df_derive(time_unit = "ns")]
    at_ns: DateTime<Utc>,
    #[df_derive(time_unit = "us")]
    o_at_us: Option<DateTime<Utc>>,
    l_at: Vec<DateTime<Utc>>,
    naive: NaiveDateTime,
    date: NaiveDate,
    l_o_date: Vec<Option<NaiveDate>>,
    time: NaiveTime,
    delta: TimeDelta,
    #[df_derive(time_unit = "ms")]
    l_delta_ms: Vec<TimeDelta>,
    elapsed: std::time::Duration,

    // Decimals.
    dec: Decimal,
    #[df_derive(decimal(precision = 20, scale = 4))]
    dec_20_4: Decimal,
    o_dec: Option<Decimal>,
    l_o_dec: Vec<Option<Decimal>>,

    // Declared encodings.
    meters: Meters,
    l_meters: Vec<Meters>,
    event: Event,
    o_event: Option<Event>,
    l_event: Vec<Event>,
    #[df_derive(with = "Cents")]
    cents: f64,
    #[df_derive(with = "Cents")]
    l_o_cents: Vec<Option<f64>>,
}

impl Gen {
    fn everything(&mut self) -> Everything<'static> {
        Everything {
            a_i8: self.int() as i8,
            a_i16: self.int() as i16,
            a_i32: self.int() as i32,
            a_i64: self.int(),
            a_i128: i128::from(self.int()) << self.below(60),
            a_u8: self.next() as u8,
            a_u16: self.next() as u16,
            a_u32: self.next() as u32,
            a_u64: self.next(),
            a_u128: u128::from(self.next()) << self.below(60),
            a_f32: self.float() as f32,
            a_f64: self.float(),
            a_bool: self.chance(50),
            a_usize: self.next() as usize,
            a_isize: self.int() as isize,
            non_zero: NonZeroU32::new(1 + self.below(1_000) as u32).unwrap(),
            unit: (),
            skipped: 0,

            text: self.text(),
            borrowed: self.word(),
            cow: if self.chance(50) {
                Cow::Borrowed(self.word())
            } else {
                Cow::Owned(self.text())
            },
            side: self.side(),
            shown: self.side(),
            bytes: self.list(|g| g.next() as u8),

            #[cfg(feature = "categorical")]
            cat_side: self.side(),
            #[cfg(feature = "categorical")]
            cat_shown: self.side(),
            #[cfg(feature = "categorical")]
            o_cat: self.option(Self::word),
            #[cfg(feature = "categorical")]
            l_o_cat: self.list(|g| g.option(Self::text)),
            #[cfg(feature = "categorical")]
            phase: self.phase(),
            #[cfg(feature = "categorical")]
            o_phase: self.option(Self::phase),
            #[cfg(feature = "categorical")]
            l_o_phase: self.list(|g| g.option(Self::phase)),

            o_i64: self.option(Self::int),
            o_bool: self.option(|g| g.chance(50)),
            o_text: self.option(Self::text),
            o_o_i32: self.option(|g| g.option(|g| g.int() as i32)),
            o_side: self.option(Self::side),
            o_bytes: self.option(|g| g.list(|g| g.next() as u8)),

            l_i32: self.list(|g| g.int() as i32),
            l_bool: self.list(|g| g.chance(50)),
            l_text: self.list(Self::text),
            l_o_i32: self.list(|g| g.option(|g| g.int() as i32)),
            l_o_bool: self.list(|g| g.option(|g| g.chance(50))),
            l_o_text: self.list(|g| g.option(Self::text)),
            l_l_i32: self.list(|g| g.list(|g| g.int() as i32)),
            l_o_l_u8: self.list(|g| g.option(|g| g.list(|g| g.next() as u8))),
            o_l_o_f64: self.option(|g| g.list(|g| g.option(Self::float))),
            l_shown: self.list(Self::side),

            boxed: Box::new(self.int()),
            o_boxed: self.option(|g| Box::new(g.text())),
            rc: Rc::new(self.float() as f32),
            arc_list: Arc::new(self.list(|g| g.next() as u32)),
            boxed_slice: self.list(|g| g.int() as i16).into_boxed_slice(),

            leaf: self.leaf(),
            o_leaf: self.option(Self::leaf),
            b_leaf: Box::new(self.leaf()),
            l_leaf: self.list(Self::leaf),
            l_o_leaf: self.list(|g| g.option(Self::leaf)),
            o_l_o_leaf: self.option(|g| g.list(|g| g.option(Self::leaf))),
            l_l_leaf: self.list(|g| {
                let len = g.below(4) as usize;
                g.list_of(len, Self::leaf)
            }),
            place: Place {
                lat: self.float() as f32,
                lon: self.float() as f32,
            },
            home: self.option(|g| Place {
                lat: g.float() as f32,
                lon: g.float() as f32,
            }),

            pair: (self.int() as i32, self.text()),
            l_pair: self.list(|g| (g.next() as u8, g.option(Self::float))),
            map: {
                let len = self.below(4) as usize;
                self.list_of(len, |g| (g.text(), g.float()))
                    .into_iter()
                    .collect()
            },
            map_of_lists: {
                let len = self.below(4) as usize;
                self.list_of(len, |g| (g.next() as u32, g.list(Self::int)))
                    .into_iter()
                    .collect()
            },
            o_map_of_leaves: self.option(|g| {
                let len = g.below(3) as usize;
                g.list_of(len, |g| (g.text(), g.leaf()))
                    .into_iter()
                    .collect()
            }),

            at: self.instant(),
            at_ns: self.instant(),
            o_at_us: self.option(Self::instant),
            l_at: self.list(Self::instant),
            naive: self.instant().naive_utc(),
            date: self.date(),
            l_o_date: self.list(|g| g.option(Self::date)),
            time: self.time(),
            delta: TimeDelta::nanoseconds(self.int()),
            l_delta_ms: self.list(|g| TimeDelta::nanoseconds(g.int())),
            elapsed: std::time::Duration::from_nanos(self.next() >> 2),

            dec: self.decimal(),
            dec_20_4: self.decimal(),
            o_dec: self.option(Self::decimal),
            l_o_dec: self.list(|g| g.option(Self::decimal)),

            meters: Meters(self.float()),
            l_meters: self.list(|g| Meters(g.float())),
            event: self.event(),
            o_event: self.option(Self::event),
            l_event: self.list(Self::event),
            cents: self.float(),
            l_o_cents: self.list(|g| g.option(Self::float)),
        }
    }
}

#[derive(ToDataFrame)]
struct Tagged<T> {
    tag: u8,
    value: T,
    more: Vec<T>,
}

/// A row small enough that a block holds thousands of them.
#[derive(ToDataFrame)]
struct Small {
    key: u32,
    flag: Option<bool>,
}

// --- agreement -----------------------------------------------------------------

#[test]
fn every_builder_agrees_with_its_definition() {
    // Rows this large come a few dozen to a block, so every length up to
    // seventy puts the end of the slice everywhere around the first two
    // boundaries, and the longer ones cross several more.
    assert!(size_of::<Everything>() >= 1024);
    let mut generator = Gen(1);
    let rows: Vec<Everything> = (0..200).map(|_| generator.everything()).collect();
    for len in (0..=70).chain([95, 96, 97, 127, 128, 129, 160, 200]) {
        check(&rows[..len]);
    }

    // One row on its own is the same row the full frame holds.
    let expected = one_at_a_time(&rows);
    for (index, row) in rows.iter().enumerate().take(24) {
        let alone = row.to_dataframe().unwrap();
        assert_same(
            &format!("row {index}"),
            &alone,
            &expected.slice(index as i64, 1),
        );
    }
}

#[test]
fn lists_longer_than_a_block_agree() {
    let mut generator = Gen(2);
    let mut rows: Vec<Everything> = (0..5).map(|_| generator.everything()).collect();
    let giant = &mut rows[2];
    giant.l_i32 = generator.list_of(9_001, |g| g.int() as i32);
    giant.l_o_i32 = generator.list_of(5_000, |g| g.option(|g| g.int() as i32));
    giant.l_bool = generator.list_of(20_000, |g| g.chance(50));
    giant.l_text = generator.list_of(1_500, Gen::text);
    giant.l_l_i32 = vec![
        Vec::new(),
        generator.list_of(4_500, |g| g.int() as i32),
        vec![7],
    ];
    giant.l_leaf = generator.list_of(700, Gen::leaf);
    giant.l_o_leaf = generator.list_of(700, |g| g.option(Gen::leaf));
    giant.l_event = generator.list_of(900, Gen::event);
    check(&rows);
}

#[test]
fn small_rows_agree_across_many_blocks() {
    let mut generator = Gen(3);
    let rows: Vec<Small> = (0..13_000)
        .map(|_| Small {
            key: generator.next() as u32,
            flag: generator.option(|g| g.chance(50)),
        })
        .collect();
    // Blocks hold a power of two of these rows. Whatever that power is, one
    // of these lengths ends a block exactly, one just before, one just after.
    for multiple in (0..=rows.len()).step_by(256) {
        for len in multiple.saturating_sub(1)..=(multiple + 1).min(rows.len()) {
            check(&rows[..len]);
        }
    }
}

#[test]
fn generic_rows_agree() {
    let mut generator = Gen(4);
    let leaves: Vec<Tagged<Leaf>> = (0..300)
        .map(|_| Tagged {
            tag: generator.next() as u8,
            value: generator.leaf(),
            more: generator.list(Gen::leaf),
        })
        .collect();
    check(&leaves);

    let floats: Vec<Tagged<Option<f64>>> = (0..3_000)
        .map(|_| Tagged {
            tag: generator.next() as u8,
            value: generator.option(Gen::float),
            more: generator.list(|g| g.option(Gen::float)),
        })
        .collect();
    check(&floats);
}

#[test]
fn floats_are_copied_bit_for_bit() {
    #[derive(ToDataFrame)]
    struct Floats {
        value: f64,
        maybe: Option<f32>,
        many: Vec<f64>,
    }

    let specials = [
        0.0,
        -0.0,
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
    ];
    let rows: Vec<Floats> = specials
        .iter()
        .cycle()
        .take(2_000)
        .enumerate()
        .map(|(index, &value)| Floats {
            value,
            maybe: (index % 3 != 0).then_some(value as f32),
            many: specials[..index % specials.len()].to_vec(),
        })
        .collect();
    let df = rows.to_dataframe().unwrap();

    let bits = |values: &Series| -> Vec<u64> {
        let values = values.f64().unwrap();
        values.into_no_null_iter().map(f64::to_bits).collect()
    };
    let value = df.column("value").unwrap().as_materialized_series();
    let expected: Vec<u64> = rows.iter().map(|row| row.value.to_bits()).collect();
    assert_eq!(bits(value), expected);
    let many = df.column("many").unwrap().list().unwrap().get_inner();
    let expected: Vec<u64> = rows
        .iter()
        .flat_map(|row| row.many.iter().map(|value| value.to_bits()))
        .collect();
    assert_eq!(bits(&many), expected);

    let maybe = df.column("maybe").unwrap().f32().unwrap().clone();
    let expected: Vec<Option<u32>> = rows.iter().map(|row| row.maybe.map(f32::to_bits)).collect();
    let actual: Vec<Option<u32>> = maybe.iter().map(|value| value.map(f32::to_bits)).collect();
    assert_eq!(actual, expected);
}

// --- errors --------------------------------------------------------------------

#[derive(ToDataFrame, Clone)]
struct Leg {
    #[df_derive(decimal(precision = 6, scale = 2))]
    price: Decimal,
    #[df_derive(time_unit = "ns")]
    at: DateTime<Utc>,
}

#[derive(ToDataFrame, Clone)]
struct Order {
    id: u64,
    #[df_derive(decimal(precision = 6, scale = 2))]
    limit: Option<Decimal>,
    legs: Vec<Leg>,
    hedge: Option<Leg>,
    quotes: BTreeMap<String, Leg>,
}

const TOO_MANY_DIGITS: &str =
    "df-derive: decimal mantissa 123456789 exceeds declared precision 6 for Decimal(6, 2)";
const TOO_FAR: &str = "df-derive: DateTime<Tz> value is out of range for nanosecond timestamps \
                       (chrono supports approximately 1677..2262)";

fn too_many_digits() -> Decimal {
    Decimal::new(123_456_789, 2)
}

fn too_far() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(3000, 1, 1, 0, 0, 0).single().unwrap()
}

fn leg() -> Leg {
    Leg {
        price: Decimal::new(12_345, 2),
        at: Utc.timestamp_opt(1_700_000_000, 0).single().unwrap(),
    }
}

fn orders(len: usize) -> Vec<Order> {
    (0..len)
        .map(|index| Order {
            id: index as u64,
            limit: (index % 2 == 0).then(|| Decimal::new(999, 1)),
            legs: vec![leg(); index % 4],
            hedge: (index % 3 == 0).then(leg),
            quotes: BTreeMap::from([("last".to_owned(), leg())]),
        })
        .collect()
}

/// Encoding `rows` fails with exactly `expected`, at the row pushing fails at.
#[track_caller]
fn assert_rejected(rows: &[Order], row: usize, expected: &str) {
    assert_eq!(first_rejected(rows), Some(row));
    let error = rows.to_dataframe().unwrap_err().to_string();
    assert_eq!(error, expected);
    let error = Order::rows_to_dataframe(rows.iter())
        .unwrap_err()
        .to_string();
    assert_eq!(error, expected);
}

#[test]
fn an_error_names_the_row_wherever_it_sits() {
    let len = 1_000;
    assert!(orders(len).to_dataframe().is_ok());
    // First and last rows, and a run of rows that crosses block boundaries.
    for row in (0..3).chain(100..400).chain(len - 3..len) {
        let mut rows = orders(len);
        rows[row].limit = Some(too_many_digits());
        assert_rejected(
            &rows,
            row,
            &format!("Order[{row}].limit: {TOO_MANY_DIGITS}"),
        );
    }
}

#[test]
fn an_error_names_the_path_to_the_value() {
    let row = 222;

    let mut rows = orders(300);
    rows[row].hedge = Some(Leg {
        at: too_far(),
        ..leg()
    });
    assert_rejected(&rows, row, &format!("Order[{row}].hedge.at: {TOO_FAR}"));

    let mut rows = orders(300);
    rows[row].legs = vec![leg(); 5];
    rows[row].legs[3].price = too_many_digits();
    assert_rejected(
        &rows,
        row,
        &format!("Order[{row}].legs[3].price: {TOO_MANY_DIGITS}"),
    );

    // An element far enough into a list to be in a later block of elements.
    let mut rows = orders(300);
    rows[row].legs = vec![leg(); 3_000];
    rows[row].legs[2_345].at = too_far();
    assert_rejected(
        &rows,
        row,
        &format!("Order[{row}].legs[2345].at: {TOO_FAR}"),
    );

    let mut rows = orders(300);
    rows[row].quotes.insert(
        "open".to_owned(),
        Leg {
            price: too_many_digits(),
            ..leg()
        },
    );
    assert_rejected(
        &rows,
        row,
        &format!("Order[{row}].quotes.value.price: {TOO_MANY_DIGITS}"),
    );
}

#[test]
fn the_first_bad_row_is_reported_not_the_first_bad_column() {
    // Columns are filled one after another, so `limit` in row 40 is reached
    // before `legs` in row 30. The error still belongs to row 30.
    let mut rows = orders(64);
    rows[40].limit = Some(too_many_digits());
    rows[30].legs = vec![leg(); 2];
    rows[30].legs[1].at = too_far();
    assert_rejected(&rows, 30, &format!("Order[30].legs[1].at: {TOO_FAR}"));

    // Within a row, fields are reported in declaration order.
    let mut rows = orders(64);
    rows[30].hedge = Some(Leg {
        price: too_many_digits(),
        at: too_far(),
    });
    assert_rejected(
        &rows,
        30,
        &format!("Order[30].hedge.price: {TOO_MANY_DIGITS}"),
    );
}

#[derive(ToDataFrame)]
#[df_derive(try_via = "ReadingRow")]
struct Reading {
    celsius: f64,
}

#[derive(ToDataFrame)]
struct ReadingRow {
    kelvin: f64,
}

impl TryFrom<&Reading> for ReadingRow {
    type Error = String;

    fn try_from(reading: &Reading) -> Result<Self, Self::Error> {
        let kelvin = reading.celsius + 273.15;
        if kelvin >= 0.0 {
            Ok(Self { kelvin })
        } else {
            Err(format!("{} °C is below absolute zero", reading.celsius))
        }
    }
}

#[test]
fn a_failed_conversion_names_the_row() {
    for row in [0, 2_047, 2_048, 2_049, 4_999] {
        let mut rows: Vec<Reading> = (0..5_000)
            .map(|index| Reading {
                celsius: f64::from(index),
            })
            .collect();
        rows[row].celsius = -300.0;
        let error = rows.to_dataframe().unwrap_err().to_string();
        assert_eq!(
            error,
            format!("Reading[{row}]: -300 °C is below absolute zero")
        );
    }
}
