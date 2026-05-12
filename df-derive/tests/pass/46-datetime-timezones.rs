use chrono::{DateTime, FixedOffset, Local, TimeZone, Utc};
use df_derive::ToDataFrame;
use polars::prelude::*;

#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

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
    #[df_derive(time_unit = "us")]
    local_us: DateTime<Local>,
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

fn datetime_value(av: AnyValue<'_>, expected_unit: TimeUnit) -> i64 {
    match av {
        AnyValue::Datetime(v, unit, _) | AnyValue::DatetimeOwned(v, unit, _) => {
            assert_eq!(unit, expected_unit, "wrong TimeUnit on AnyValue");
            v
        }
        other => panic!("unexpected Datetime AnyValue: {other:?}"),
    }
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
            AnyValue::Datetime(v, actual_unit, _) | AnyValue::DatetimeOwned(v, actual_unit, _) => {
                assert_eq!(actual_unit, unit, "wrong TimeUnit inside list {col}");
                Some(v)
            }
            AnyValue::Null => None,
            other => panic!("unexpected AnyValue inside list {col}: {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected, "col {col} row {row}");
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
        local_us: local,
    };
    let missing = ZonedTimes {
        fixed_default: fixed_later,
        fixed_us: fixed_later,
        fixed_ns: fixed_later,
        maybe_fixed: None,
        fixed_log: vec![],
        nullable_fixed_log: vec![None],
        local_default: fixed_later.with_timezone(&Local),
        local_us: fixed_later.with_timezone(&Local),
    };

    let schema = ZonedTimes::schema().unwrap();
    assert_eq!(
        schema_dtype(&schema, "fixed_default"),
        DataType::Datetime(TimeUnit::Milliseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "fixed_us"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "fixed_ns"),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "fixed_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Milliseconds, None)))
    );
    assert_eq!(
        schema_dtype(&schema, "nullable_fixed_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Milliseconds, None)))
    );
    assert_eq!(
        schema_dtype(&schema, "local_default"),
        DataType::Datetime(TimeUnit::Milliseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "local_us"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );

    let df = row.clone().to_dataframe().unwrap();
    assert_eq!(
        dtype(&df, "fixed_default"),
        DataType::Datetime(TimeUnit::Milliseconds, None)
    );
    assert_eq!(
        dtype(&df, "fixed_us"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        dtype(&df, "local_default"),
        DataType::Datetime(TimeUnit::Milliseconds, None)
    );
    assert_datetime(
        &df,
        "fixed_default",
        TimeUnit::Milliseconds,
        base.timestamp_millis(),
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
        TimeUnit::Milliseconds,
        later.timestamp_millis(),
    );
    assert_datetime(
        &df,
        "local_default",
        TimeUnit::Milliseconds,
        base.timestamp_millis(),
    );
    assert_datetime(&df, "local_us", TimeUnit::Microseconds, base.timestamp_micros());
    assert_datetime_list(
        &df,
        "fixed_log",
        0,
        TimeUnit::Milliseconds,
        &[Some(base.timestamp_millis()), Some(later.timestamp_millis())],
    );
    assert_datetime_list(
        &df,
        "nullable_fixed_log",
        0,
        TimeUnit::Milliseconds,
        &[Some(base.timestamp_millis()), None, Some(later.timestamp_millis())],
    );

    let batch = vec![row, missing];
    let batch_df = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.height(), 2);
    assert_datetime(
        &batch_df,
        "fixed_default",
        TimeUnit::Milliseconds,
        base.timestamp_millis(),
    );
    assert_datetime_list(
        &batch_df,
        "fixed_log",
        1,
        TimeUnit::Milliseconds,
        &[],
    );
    assert_datetime_list(
        &batch_df,
        "nullable_fixed_log",
        1,
        TimeUnit::Milliseconds,
        &[None],
    );
    assert!(matches!(
        batch_df.column("maybe_fixed").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
}
