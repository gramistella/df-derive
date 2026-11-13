// A `DateTime<Tz>` is an instant, and its column is marked as UTC. A
// `NaiveDateTime` names no zone, and neither does its column. Both count
// microseconds unless the field chooses another unit. The tests here state
// those rules once, for every place a datetime can sit.

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
    #[df_derive(time_unit = "ms")]
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
        ("at", instant(Microseconds)),
        ("at_ns", instant(Nanoseconds)),
        ("offset", instant(Microseconds)),
        ("maybe", instant(Microseconds)),
        ("log", list(instant(Milliseconds))),
        ("boxed", instant(Microseconds)),
        ("by_instant.key", list(instant(Microseconds))),
        ("by_instant.value", list(instant(Microseconds))),
        ("wall", naive(Microseconds)),
        ("wall_ns", naive(Nanoseconds)),
        ("walls", list(naive(Microseconds))),
        ("stamp.at", instant(Microseconds)),
        ("stamp.wall", naive(Microseconds)),
        ("stamps.at", list(instant(Microseconds))),
        ("stamps.wall", list(naive(Microseconds))),
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
    let utc = row.at.timestamp_micros();
    let df = row.to_dataframe().unwrap();
    let first = |name: &str| counts(df.column(name).unwrap())[0].unwrap();

    // Noon at +02:00 and ten o'clock UTC are one instant: one stored count,
    // and a column that says UTC for both spellings of it.
    assert_eq!(first("offset"), utc);
    assert_eq!(first("at"), utc);
    assert_eq!(
        df.column("offset").unwrap().get(0).unwrap(),
        AnyValue::Datetime(utc, TimeUnit::Microseconds, Some(&TimeZone::UTC))
    );

    // A wall-clock time is stored as written, with no zone to convert from:
    // noon stays noon, two hours after the instant's reading in UTC.
    assert_eq!(first("wall") - utc, 2 * 3_600_000_000);
    assert_eq!(
        df.column("wall").unwrap().get(0).unwrap(),
        AnyValue::Datetime(utc + 2 * 3_600_000_000, TimeUnit::Microseconds, None)
    );
}

#[test]
fn casting_drops_the_zone_and_keeps_the_values() {
    // Code that needs a zone-less column casts the UTC one. The stored counts
    // are the same before and after.
    let rows: Vec<Times> = (1..=9).map(times).collect();
    let df = rows.to_dataframe().unwrap();

    for (name, unit) in [
        ("at", TimeUnit::Microseconds),
        ("at_ns", TimeUnit::Nanoseconds),
        ("maybe", TimeUnit::Microseconds),
    ] {
        let zoned = df.column(name).unwrap();
        assert_eq!(zoned.dtype(), &instant(unit));
        let zoneless = zoned.cast(&naive(unit)).unwrap();
        assert_eq!(zoneless.dtype(), &naive(unit));
        assert_eq!(counts(&zoneless), counts(zoned));
    }
    assert_eq!(df.column("maybe").unwrap().null_count(), 3);
}

// --- what the default unit keeps, and what it holds --------------------------------

#[derive(ToDataFrame)]
struct Tick {
    at: DateTime<Utc>,
    wall: NaiveDateTime,
    #[df_derive(time_unit = "ms")]
    coarse: DateTime<Utc>,
    #[df_derive(time_unit = "ns")]
    fine: DateTime<Utc>,
}

#[test]
fn the_default_unit_keeps_microseconds() {
    // 1.234567891 seconds after the epoch.
    let at = DateTime::<Utc>::from_timestamp(1, 234_567_891).unwrap();
    let row = Tick {
        at,
        wall: at.naive_utc(),
        coarse: at,
        fine: at,
    };
    let df = row.to_dataframe().unwrap();
    let first = |name: &str| counts(df.column(name).unwrap())[0].unwrap();

    // Digits finer than the unit are dropped, whatever the unit.
    assert_eq!(first("at"), 1_234_567);
    assert_eq!(first("wall"), 1_234_567);
    assert_eq!(first("coarse"), 1_234);
    assert_eq!(first("fine"), 1_234_567_891);

    // The digits are those of the date, so before 1970 the count moves away
    // from zero: 23:59:59.998_765_4 on the last day of 1969.
    let at = DateTime::<Utc>::from_timestamp(-1, 998_765_400).unwrap();
    let row = Tick {
        at,
        wall: at.naive_utc(),
        coarse: at,
        fine: at,
    };
    let df = row.to_dataframe().unwrap();
    let first = |name: &str| counts(df.column(name).unwrap())[0].unwrap();
    assert_eq!(first("at"), -1_235);
    assert_eq!(first("wall"), -1_235);
    assert_eq!(first("coarse"), -2);
    assert_eq!(first("fine"), -1_234_600);
}

#[derive(ToDataFrame)]
struct Far {
    at: DateTime<Utc>,
    wall: NaiveDateTime,
}

#[derive(ToDataFrame)]
struct FarInNanoseconds {
    #[df_derive(time_unit = "ns")]
    at: DateTime<Utc>,
}

#[test]
fn the_default_unit_holds_every_date_chrono_has() {
    // The ends of chrono's range, and two dates data sets use to mean "never".
    let year = |year: i32| {
        NaiveDate::from_ymd_opt(year, 12, 31)
            .unwrap()
            .and_hms_opt(23, 59, 59)
            .unwrap()
            .and_utc()
    };
    for at in [
        DateTime::<Utc>::MIN_UTC,
        year(1),
        year(9999),
        DateTime::<Utc>::MAX_UTC,
    ] {
        let df = Far {
            at,
            wall: at.naive_utc(),
        }
        .to_dataframe()
        .unwrap();
        assert_eq!(
            counts(df.column("at").unwrap()),
            [Some(at.timestamp_micros())]
        );
        assert_eq!(
            counts(df.column("wall").unwrap()),
            [Some(at.timestamp_micros())]
        );
    }

    // Nanoseconds reach from 1677 to 2262. A field that asks for them says so
    // when a value is outside.
    let error = FarInNanoseconds { at: year(9999) }
        .to_dataframe()
        .unwrap_err()
        .to_string();
    assert!(error.contains("out of range for nanosecond"), "{error}");
}
