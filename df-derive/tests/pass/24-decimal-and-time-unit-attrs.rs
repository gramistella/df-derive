use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};

use chrono::{DateTime, TimeZone, Utc};
use polars::prelude::*;
use rust_decimal::Decimal;

// Every wrapper shape (`T`, `Option<T>`, `Vec<T>`, `Option<Vec<T>>`,
// `Vec<Option<T>>`) under each attribute. The attribute applies to the
// innermost value, and the dtype assertions below check that it does.
#[derive(ToDataFrame, Clone)]
struct OverrideEverything {
    #[df_derive(decimal(precision = 18, scale = 6))]
    price: Decimal,
    #[df_derive(decimal(precision = 38, scale = 18))]
    big_amount: Option<Decimal>,
    #[df_derive(decimal(precision = 12, scale = 4))]
    fees: Vec<Decimal>,
    #[df_derive(decimal(precision = 30, scale = 12))]
    opt_amounts: Option<Vec<Decimal>>,
    #[df_derive(decimal(precision = 16, scale = 5))]
    nullable_fees: Vec<Option<Decimal>>,

    #[df_derive(time_unit = "ms")]
    ts_ms: DateTime<Utc>,
    #[df_derive(time_unit = "us")]
    ts_us: DateTime<Utc>,
    #[df_derive(time_unit = "ns")]
    ts_ns: DateTime<Utc>,
    #[df_derive(time_unit = "us")]
    opt_ts: Option<DateTime<Utc>>,
    #[df_derive(time_unit = "ns")]
    ts_log: Vec<DateTime<Utc>>,
    #[df_derive(time_unit = "ns")]
    nullable_ts_log: Vec<Option<DateTime<Utc>>>,

    // No override → default behavior preserved.
    default_price: Decimal,
    default_ts: DateTime<Utc>,
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

/// The column type of a `DateTime<Tz>` field. The value is an instant, so the
/// column is marked as UTC.
fn instant(unit: TimeUnit) -> DataType {
    DataType::Datetime(unit, Some(polars::prelude::TimeZone::UTC))
}

fn main() {
    let ts = Utc.timestamp_micros(1_700_000_000_123_456).single().unwrap();

    let row = OverrideEverything {
        price: Decimal::new(12345, 2),
        big_amount: Some(Decimal::new(1, 18)),
        fees: vec![Decimal::new(50, 2), Decimal::new(75, 2)],
        opt_amounts: Some(vec![Decimal::new(123, 4), Decimal::new(456, 4)]),
        nullable_fees: vec![Some(Decimal::new(100, 2)), None, Some(Decimal::new(200, 2))],

        ts_ms: ts,
        ts_us: ts,
        ts_ns: ts,
        opt_ts: Some(ts),
        ts_log: vec![ts, ts],
        nullable_ts_log: vec![Some(ts), None, Some(ts)],

        default_price: Decimal::new(99, 2),
        default_ts: ts,
    };

    // The schema shows each field's attribute exactly. A failure here means an
    // attribute did not reach the builder of the innermost value, or the
    // default was used in its place.
    let schema = OverrideEverything::schema().unwrap();
    assert_eq!(schema_dtype(&schema, "price"), DataType::Decimal(18, 6));
    assert_eq!(schema_dtype(&schema, "big_amount"), DataType::Decimal(38, 18));
    assert_eq!(
        schema_dtype(&schema, "fees"),
        DataType::List(Box::new(DataType::Decimal(12, 4)))
    );
    assert_eq!(
        schema_dtype(&schema, "opt_amounts"),
        DataType::List(Box::new(DataType::Decimal(30, 12)))
    );
    assert_eq!(
        schema_dtype(&schema, "nullable_fees"),
        DataType::List(Box::new(DataType::Decimal(16, 5)))
    );
    assert_eq!(
        schema_dtype(&schema, "ts_ms"),
        instant(TimeUnit::Milliseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "ts_us"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "ts_ns"),
        instant(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "opt_ts"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "ts_log"),
        DataType::List(Box::new(instant(TimeUnit::Nanoseconds)))
    );
    assert_eq!(
        schema_dtype(&schema, "nullable_ts_log"),
        DataType::List(Box::new(instant(TimeUnit::Nanoseconds)))
    );
    assert_eq!(
        schema_dtype(&schema, "default_price"),
        DataType::Decimal(38, 10)
    );
    assert_eq!(
        schema_dtype(&schema, "default_ts"),
        instant(TimeUnit::Microseconds)
    );

    // Single-row materialization preserves the schema dtypes.
    let df = row.to_dataframe().unwrap();
    assert_eq!(dtype(&df, "price"), DataType::Decimal(18, 6));
    assert_eq!(dtype(&df, "big_amount"), DataType::Decimal(38, 18));
    assert_eq!(
        dtype(&df, "fees"),
        DataType::List(Box::new(DataType::Decimal(12, 4)))
    );
    assert_eq!(
        dtype(&df, "opt_amounts"),
        DataType::List(Box::new(DataType::Decimal(30, 12)))
    );
    assert_eq!(
        dtype(&df, "nullable_fees"),
        DataType::List(Box::new(DataType::Decimal(16, 5)))
    );
    assert_eq!(
        dtype(&df, "ts_ms"),
        instant(TimeUnit::Milliseconds)
    );
    assert_eq!(
        dtype(&df, "ts_us"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        dtype(&df, "ts_ns"),
        instant(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        dtype(&df, "opt_ts"),
        instant(TimeUnit::Microseconds)
    );
    assert_eq!(
        dtype(&df, "ts_log"),
        DataType::List(Box::new(instant(TimeUnit::Nanoseconds)))
    );
    assert_eq!(
        dtype(&df, "nullable_ts_log"),
        DataType::List(Box::new(instant(TimeUnit::Nanoseconds)))
    );
    assert_eq!(dtype(&df, "default_price"), DataType::Decimal(38, 10));
    assert_eq!(
        dtype(&df, "default_ts"),
        instant(TimeUnit::Microseconds)
    );

    // More than one row must produce the same dtypes.
    let batch = vec![row.clone(), row.clone()];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.height(), 2);
    assert_eq!(dtype(&df_batch, "price"), DataType::Decimal(18, 6));
    assert_eq!(
        dtype(&df_batch, "ts_ns"),
        instant(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        dtype(&df_batch, "ts_log"),
        DataType::List(Box::new(instant(TimeUnit::Nanoseconds)))
    );
    assert_eq!(
        dtype(&df_batch, "fees"),
        DataType::List(Box::new(DataType::Decimal(12, 4)))
    );
    assert_eq!(
        dtype(&df_batch, "nullable_fees"),
        DataType::List(Box::new(DataType::Decimal(16, 5)))
    );
    assert_eq!(
        dtype(&df_batch, "nullable_ts_log"),
        DataType::List(Box::new(instant(TimeUnit::Nanoseconds)))
    );

    // The unit decides the number that is stored, not only the type of the
    // column: the same instant is a count 1,000 times larger in each finer
    // unit, and a coarser unit drops the digits it cannot hold. A field
    // without the attribute stores microseconds.
    fn datetime_value(av: AnyValue<'_>, expected_unit: TimeUnit) -> i64 {
        match av {
            AnyValue::Datetime(v, u, _) | AnyValue::DatetimeOwned(v, u, _) => {
                assert_eq!(u, expected_unit, "wrong TimeUnit on AnyValue");
                v
            }
            other => panic!("unexpected Datetime AnyValue: {other:?}"),
        }
    }
    let stored = |col: &str, unit| datetime_value(df.column(col).unwrap().get(0).unwrap(), unit);
    assert_eq!(stored("ts_ms", TimeUnit::Milliseconds), 1_700_000_000_123);
    assert_eq!(stored("ts_us", TimeUnit::Microseconds), 1_700_000_000_123_456);
    assert_eq!(
        stored("ts_ns", TimeUnit::Nanoseconds),
        1_700_000_000_123_456_000
    );
    assert_eq!(
        stored("default_ts", TimeUnit::Microseconds),
        1_700_000_000_123_456
    );

    // Empty dataframe also reports the override dtypes.
    let empty = OverrideEverything::empty_dataframe().unwrap();
    assert_eq!(dtype(&empty, "ts_ns"), instant(TimeUnit::Nanoseconds));
    assert_eq!(dtype(&empty, "price"), DataType::Decimal(18, 6));
}
