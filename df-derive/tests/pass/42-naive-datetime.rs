use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};

use chrono::{Duration as ChronoDuration, NaiveDate, NaiveDateTime};
use polars::prelude::*;

#[derive(ToDataFrame, Clone)]
struct NaiveDateTimeRow {
    default_dt: NaiveDateTime,
    optional_dt: Option<NaiveDateTime>,
    dt_log: Vec<NaiveDateTime>,
    nullable_dt_log: Vec<Option<NaiveDateTime>>,
    maybe_dt_log: Option<Vec<NaiveDateTime>>,
    #[df_derive(time_unit = "ms")]
    millis: NaiveDateTime,
    #[df_derive(time_unit = "us")]
    micros: NaiveDateTime,
    #[df_derive(time_unit = "ns")]
    nanos: NaiveDateTime,
    tuple_dt: (NaiveDateTime, NaiveDateTime),
}

fn dtype(df: &DataFrame, col: &str) -> DataType {
    df.column(col).unwrap().dtype().clone()
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(n, _)| n == col)
        .map(|(_, dt)| dt.clone())
        .unwrap_or_else(|| panic!("column {col} missing from schema"))
}

fn datetime_value(av: AnyValue<'_>, expected_unit: TimeUnit) -> Option<i64> {
    match av {
        AnyValue::Datetime(v, u, _) | AnyValue::DatetimeOwned(v, u, _) => {
            assert_eq!(u, expected_unit, "wrong TimeUnit on Datetime AnyValue");
            Some(v)
        }
        AnyValue::Null => None,
        other => panic!("expected Datetime AnyValue or Null, got {other:?}"),
    }
}

fn assert_list_datetimes(
    df: &DataFrame,
    col: &str,
    row: usize,
    expected: &[Option<i64>],
    expected_unit: TimeUnit,
) {
    let v = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected list at {col}[{row}], got {v:?}");
    };
    let actual: Vec<Option<i64>> = inner
        .iter()
        .map(|av| datetime_value(av, expected_unit))
        .collect();
    assert_eq!(actual, expected, "{col}[{row}] mismatch");
}

fn main() {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_nano_opt(0, 0, 0, 0)
        .unwrap();
    let plus_ms = epoch + ChronoDuration::milliseconds(1_234);
    let minus_ms = epoch - ChronoDuration::milliseconds(2);
    let plus_us = epoch + ChronoDuration::microseconds(987_654);
    let plus_ns = epoch + ChronoDuration::nanoseconds(1_234);

    let row = NaiveDateTimeRow {
        default_dt: plus_ms,
        optional_dt: Some(minus_ms),
        dt_log: vec![epoch, plus_ms],
        nullable_dt_log: vec![Some(plus_ms), None, Some(minus_ms)],
        maybe_dt_log: Some(vec![minus_ms, epoch]),
        millis: plus_us,
        micros: plus_us,
        nanos: plus_ns,
        tuple_dt: (epoch, plus_ms),
    };

    let schema = NaiveDateTimeRow::schema().unwrap();
    assert_eq!(
        schema_dtype(&schema, "default_dt"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "optional_dt"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "dt_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Microseconds, None)))
    );
    assert_eq!(
        schema_dtype(&schema, "nullable_dt_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Microseconds, None)))
    );
    assert_eq!(
        schema_dtype(&schema, "maybe_dt_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Microseconds, None)))
    );
    assert_eq!(
        schema_dtype(&schema, "millis"),
        DataType::Datetime(TimeUnit::Milliseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "micros"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "nanos"),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "tuple_dt.field_0"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "tuple_dt.field_1"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );

    let df = row.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 10));
    assert_eq!(
        dtype(&df, "default_dt"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        datetime_value(df.column("default_dt").unwrap().get(0).unwrap(), TimeUnit::Microseconds),
        Some(plus_ms.and_utc().timestamp_micros())
    );
    assert_eq!(
        datetime_value(df.column("optional_dt").unwrap().get(0).unwrap(), TimeUnit::Microseconds),
        Some(minus_ms.and_utc().timestamp_micros())
    );
    // 0.987654 s in a coarser unit: the digits finer than the unit are dropped.
    assert_eq!(
        datetime_value(df.column("millis").unwrap().get(0).unwrap(), TimeUnit::Milliseconds),
        Some(987)
    );
    assert_eq!(
        datetime_value(df.column("micros").unwrap().get(0).unwrap(), TimeUnit::Microseconds),
        Some(987_654)
    );
    assert_eq!(
        datetime_value(df.column("nanos").unwrap().get(0).unwrap(), TimeUnit::Nanoseconds),
        plus_ns.and_utc().timestamp_nanos_opt()
    );
    assert_eq!(
        datetime_value(
            df.column("tuple_dt.field_0").unwrap().get(0).unwrap(),
            TimeUnit::Microseconds,
        ),
        Some(epoch.and_utc().timestamp_micros())
    );
    assert_eq!(
        datetime_value(
            df.column("tuple_dt.field_1").unwrap().get(0).unwrap(),
            TimeUnit::Microseconds,
        ),
        Some(plus_ms.and_utc().timestamp_micros())
    );
    assert_list_datetimes(
        &df,
        "dt_log",
        0,
        &[
            Some(epoch.and_utc().timestamp_micros()),
            Some(plus_ms.and_utc().timestamp_micros()),
        ],
        TimeUnit::Microseconds,
    );
    assert_list_datetimes(
        &df,
        "nullable_dt_log",
        0,
        &[
            Some(plus_ms.and_utc().timestamp_micros()),
            None,
            Some(minus_ms.and_utc().timestamp_micros()),
        ],
        TimeUnit::Microseconds,
    );

    let batch = vec![
        row.clone(),
        NaiveDateTimeRow {
            default_dt: epoch,
            optional_dt: None,
            dt_log: vec![],
            nullable_dt_log: vec![None],
            maybe_dt_log: None,
            millis: epoch,
            micros: epoch,
            nanos: epoch,
            tuple_dt: (minus_ms, epoch),
        },
    ];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (2, 10));
    assert_eq!(
        datetime_value(
            df_batch.column("optional_dt").unwrap().get(1).unwrap(),
            TimeUnit::Microseconds,
        ),
        None
    );
    assert_list_datetimes(&df_batch, "dt_log", 1, &[], TimeUnit::Microseconds);
    assert!(matches!(
        df_batch.column("maybe_dt_log").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));

    let empty = NaiveDateTimeRow::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 10));
    assert_eq!(
        dtype(&empty, "default_dt"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        dtype(&empty, "nanos"),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
}
