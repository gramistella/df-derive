use chrono::{DateTime, FixedOffset, Local, TimeZone, Utc};
use df_derive::ToDataFrame;
use polars::prelude::*;

use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
struct ZonedTimes {
    fixed_default: DateTime<FixedOffset>,
    #[df_derive(time_unit = "us")]
    fixed_us: DateTime<FixedOffset>,
    #[df_derive(time_unit = "ns")]
    fixed_ns: DateTime<FixedOffset>,
    maybe_fixed: Option<DateTime<FixedOffset>>,
    fixed_log: Vec<DateTime<FixedOffset>>,
    nullable_fixed_log: Vec<Option<DateTime<FixedOffset>>>,
    local_default: DateTime<Local>,
    #[df_derive(time_unit = "ms")]
    local_ms: DateTime<Local>,
}

fn dtype(df: &DataFrame, col: &str) -> DataType {
    df.column(col).unwrap().dtype().clone()
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(name, _)| name == col)
        .map(|(_, dtype)| dtype.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}

/// The count inside a datetime value, after checking its unit and that the
/// value is marked as UTC whatever zone or offset it was built with.
fn datetime_value(av: AnyValue<'_>, expected_unit: TimeUnit) -> i64 {
    let (value, unit, zone) = match av {
        AnyValue::Datetime(value, unit, zone) => (value, unit, zone.cloned()),
        AnyValue::DatetimeOwned(value, unit, zone) => {
            (value, unit, zone.map(|zone| (*zone).clone()))
        }
        other => panic!("unexpected Datetime AnyValue: {other:?}"),
    };
    assert_eq!(unit, expected_unit, "wrong TimeUnit on AnyValue");
    assert_eq!(zone, Some(polars::prelude::TimeZone::UTC), "not marked as UTC");
    value
}

fn assert_datetime(df: &DataFrame, col: &str, unit: TimeUnit, expected: i64) {
    let value = datetime_value(df.column(col).unwrap().get(0).unwrap(), unit);
    assert_eq!(value, expected, "col {col}");
}

fn assert_datetime_list(
    df: &DataFrame,
    col: &str,
    row: usize,
    unit: TimeUnit,
    expected: &[Option<i64>],
) {
    let v = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected List for {col} row {row}, got {v:?}");
    };
    let actual: Vec<Option<i64>> = inner
        .iter()
        .map(|av| match av {
            AnyValue::Null => None,
            value => Some(datetime_value(value, unit)),
        })
        .collect();
    assert_eq!(actual, expected, "col {col} row {row}");
}

/// The column type of a `DateTime<Tz>` field. The value is an instant, so the
/// column is marked as UTC.
fn instant(unit: TimeUnit) -> DataType {
    DataType::Datetime(unit, Some(polars::prelude::TimeZone::UTC))
}

fn main() {
    println!("--- chrono::DateTime<Tz> support ---");

    let base = Utc.timestamp_millis_opt(1_700_000_000_123).single().unwrap();
    let later = Utc.timestamp_millis_opt(1_700_000_001_456).single().unwrap();
    let offset = FixedOffset::east_opt(2 * 3600 + 30 * 60).unwrap();
    let fixed = base.with_timezone(&offset);
    let fixed_later = later.with_timezone(&offset);
    let local = base.with_timezone(&Local);

    let row = ZonedTimes {
        fixed_default: fixed,
        fixed_us: fixed,
        fixed_ns: fixed,
        maybe_fixed: Some(fixed_later),
        fixed_log: vec![fixed, fixed_later],
        nullable_fixed_log: vec![Some(fixed), None, Some(fixed_later)],
        local_default: local,
        local_ms: local,
    };
    let missing = ZonedTimes {
        fixed_default: fixed_later,
        fixed_us: fixed_later,
        fixed_ns: fixed_later,
        maybe_fixed: None,
        fixed_log: vec![],
        nullable_fixed_log: vec![None],
        local_default: fixed_later.with_timezone(&Local),
        local_ms: fixed_later.with_timezone(&Local),
    };

    let schema = ZonedTimes::schema().unwrap();
    assert_eq!(
        schema_dtype(&schema, "fixed_default"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "fixed_us"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "fixed_ns"),
        instant(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "fixed_log"),
        DataType::List(Box::new(instant(TimeUnit::Microseconds)))
    );
    assert_eq!(
        schema_dtype(&schema, "nullable_fixed_log"),
        DataType::List(Box::new(instant(TimeUnit::Microseconds)))
    );
    assert_eq!(
        schema_dtype(&schema, "local_default"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "local_ms"),
        instant(TimeUnit::Milliseconds)
    );

    let df = row.clone().to_dataframe().unwrap();
    assert_eq!(
        dtype(&df, "fixed_default"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        dtype(&df, "fixed_us"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        dtype(&df, "local_default"),
        instant(TimeUnit::Microseconds)
    );
    assert_datetime(
        &df,
        "fixed_default",
        TimeUnit::Microseconds,
        base.timestamp_micros(),
    );
    assert_datetime(
        &df,
        "fixed_us",
        TimeUnit::Microseconds,
        base.timestamp_micros(),
    );
    assert_datetime(
        &df,
        "fixed_ns",
        TimeUnit::Nanoseconds,
        base.timestamp_nanos_opt().unwrap(),
    );
    assert_datetime(
        &df,
        "maybe_fixed",
        TimeUnit::Microseconds,
        later.timestamp_micros(),
    );
    assert_datetime(
        &df,
        "local_default",
        TimeUnit::Microseconds,
        base.timestamp_micros(),
    );
    assert_datetime(&df, "local_ms", TimeUnit::Milliseconds, base.timestamp_millis());
    assert_datetime_list(
        &df,
        "fixed_log",
        0,
        TimeUnit::Microseconds,
        &[Some(base.timestamp_micros()), Some(later.timestamp_micros())],
    );
    assert_datetime_list(
        &df,
        "nullable_fixed_log",
        0,
        TimeUnit::Microseconds,
        &[Some(base.timestamp_micros()), None, Some(later.timestamp_micros())],
    );

    let batch = vec![row, missing];
    let batch_df = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.height(), 2);
    assert_datetime(
        &batch_df,
        "fixed_default",
        TimeUnit::Microseconds,
        base.timestamp_micros(),
    );
    assert_datetime_list(
        &batch_df,
        "fixed_log",
        1,
        TimeUnit::Microseconds,
        &[],
    );
    assert_datetime_list(
        &batch_df,
        "nullable_fixed_log",
        1,
        TimeUnit::Microseconds,
        &[None],
    );
    assert!(matches!(
        batch_df.column("maybe_fixed").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
}
