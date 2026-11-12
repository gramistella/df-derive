// A `DateTime<Tz>` is an instant, and its column is marked as UTC. A
// `NaiveDateTime` names no zone, and neither does its column. The tests here
// state that rule once, for every place a datetime can sit.

#![allow(dead_code)]

use std::collections::BTreeMap;

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, TimeZone as _, Utc};
use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};
use polars::prelude::*;

fn instant(unit: TimeUnit) -> DataType {
    DataType::Datetime(unit, Some(TimeZone::UTC))
}

fn naive(unit: TimeUnit) -> DataType {
    DataType::Datetime(unit, None)
}

fn list(inner: DataType) -> DataType {
    DataType::List(Box::new(inner))
}

#[derive(ToDataFrame, Clone)]
struct Stamp {
    at: DateTime<Utc>,
    wall: NaiveDateTime,
}

#[derive(ToDataFrame, Clone)]
struct Times {
    at: DateTime<Utc>,
    #[df_derive(time_unit = "ns")]
    at_ns: DateTime<Utc>,
    offset: DateTime<FixedOffset>,
    maybe: Option<DateTime<Utc>>,
    #[df_derive(time_unit = "us")]
    log: Vec<Option<DateTime<Utc>>>,
    boxed: Box<DateTime<Utc>>,
    by_instant: BTreeMap<DateTime<Utc>, DateTime<FixedOffset>>,
    wall: NaiveDateTime,
    #[df_derive(time_unit = "ns")]
    wall_ns: Option<NaiveDateTime>,
    walls: Vec<NaiveDateTime>,
    stamp: Stamp,
    stamps: Vec<Stamp>,
}

/// Noon on a day in March 2024, as a wall-clock time.
fn noon(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 3, day)
        .unwrap()
        .and_hms_opt(12, 0, 0)
        .unwrap()
}

/// Two hours east of UTC.
fn plus_two() -> FixedOffset {
    FixedOffset::east_opt(2 * 3_600).unwrap()
}

/// A row whose instants all fall on `day`. Every third row has no optional
/// values and empty lists.
fn times(day: u32) -> Times {
    let wall = noon(day);
    let offset = plus_two().from_local_datetime(&wall).unwrap();
    let at = offset.with_timezone(&Utc);
    let sparse = day.is_multiple_of(3);
    let stamp = Stamp { at, wall };
    Times {
        at,
        at_ns: at,
        offset,
        maybe: (!sparse).then_some(at),
        log: if sparse {
            Vec::new()
        } else {
            vec![Some(at), None]
        },
        boxed: Box::new(at),
        by_instant: if sparse {
            BTreeMap::new()
        } else {
            BTreeMap::from([(at, offset)])
        },
        wall,
        wall_ns: (!sparse).then_some(wall),
        walls: if sparse { Vec::new() } else { vec![wall] },
        stamp: stamp.clone(),
        stamps: if sparse { Vec::new() } else { vec![stamp] },
    }
}

fn dtypes(df: &DataFrame) -> Vec<(String, DataType)> {
    df.columns()
        .iter()
        .map(|column| (column.name().to_string(), column.dtype().clone()))
        .collect()
}

/// The stored counts of a datetime column.
fn counts(column: &Column) -> Vec<Option<i64>> {
    column.datetime().unwrap().physical().iter().collect()
}

#[test]
fn an_instant_column_is_utc_and_a_naive_one_has_no_zone() {
    use TimeUnit::{Microseconds, Milliseconds, Nanoseconds};

    let expected: Vec<(String, DataType)> = [
        ("at", instant(Milliseconds)),
        ("at_ns", instant(Nanoseconds)),
        ("offset", instant(Milliseconds)),
        ("maybe", instant(Milliseconds)),
        ("log", list(instant(Microseconds))),
        ("boxed", instant(Milliseconds)),
        ("by_instant.key", list(instant(Milliseconds))),
        ("by_instant.value", list(instant(Milliseconds))),
        ("wall", naive(Milliseconds)),
        ("wall_ns", naive(Nanoseconds)),
        ("walls", list(naive(Milliseconds))),
        ("stamp.at", instant(Milliseconds)),
        ("stamp.wall", naive(Milliseconds)),
        ("stamps.at", list(instant(Milliseconds))),
        ("stamps.wall", list(naive(Milliseconds))),
    ]
    .into_iter()
    .map(|(name, dtype)| (name.to_owned(), dtype))
    .collect();

    assert_eq!(Times::schema().unwrap(), expected);
    // The frames agree with the schema: with no rows, with one, and with
    // enough rows that some are sparse and some are not.
    assert_eq!(dtypes(&Times::empty_dataframe().unwrap()), expected);
    assert_eq!(dtypes(&times(1).to_dataframe().unwrap()), expected);
    let rows: Vec<Times> = (1..=28).map(times).collect();
    assert_eq!(dtypes(&rows.to_dataframe().unwrap()), expected);
}

#[test]
fn an_offset_is_not_kept_but_the_instant_is() {
    let row = times(1);
    let utc = row.at.timestamp_millis();
    let df = row.to_dataframe().unwrap();
    let first = |name: &str| counts(df.column(name).unwrap())[0].unwrap();

    // Noon at +02:00 and ten o'clock UTC are one instant: one stored count,
    // and a column that says UTC for both spellings of it.
    assert_eq!(first("offset"), utc);
    assert_eq!(first("at"), utc);
    assert_eq!(
        df.column("offset").unwrap().get(0).unwrap(),
        AnyValue::Datetime(utc, TimeUnit::Milliseconds, Some(&TimeZone::UTC))
    );

    // A wall-clock time is stored as written, with no zone to convert from:
    // noon stays noon, two hours after the instant's reading in UTC.
    assert_eq!(first("wall") - utc, 2 * 3_600_000);
    assert_eq!(
        df.column("wall").unwrap().get(0).unwrap(),
        AnyValue::Datetime(utc + 2 * 3_600_000, TimeUnit::Milliseconds, None)
    );
}

#[test]
fn casting_drops_the_zone_and_keeps_the_values() {
    // Code that needs a zone-less column casts the UTC one. The stored counts
    // are the same before and after.
    let rows: Vec<Times> = (1..=9).map(times).collect();
    let df = rows.to_dataframe().unwrap();

    for (name, unit) in [
        ("at", TimeUnit::Milliseconds),
        ("at_ns", TimeUnit::Nanoseconds),
        ("maybe", TimeUnit::Milliseconds),
    ] {
        let zoned = df.column(name).unwrap();
        assert_eq!(zoned.dtype(), &instant(unit));
        let zoneless = zoned.cast(&naive(unit)).unwrap();
        assert_eq!(zoneless.dtype(), &naive(unit));
        assert_eq!(counts(&zoneless), counts(zoned));
    }
    assert_eq!(df.column("maybe").unwrap().null_count(), 3);
}
