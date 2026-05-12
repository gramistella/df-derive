use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

use chrono::{DateTime, TimeZone, Utc};
use polars::prelude::*;
use rust_decimal::Decimal;

// Mix every supported wrapper shape (`T`, `Option<T>`, `Vec<T>`,
// `Option<Vec<T>>`, `Vec<Option<T>>`) so a regression in any of the wrapper-
// aware codegen paths is caught by the schema/runtime dtype assertions below.
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

fn main() {
    let ts = Utc.timestamp_millis_opt(1_700_000_000_123).single().unwrap();

    let row = OverrideEverything {
        price: Decimal::new(12345, 2),
        big_amount: Some(Decimal::new(1, 18)),
        fees: vec![Decimal::new(50, 2), Decimal::new(75, 2)],
        opt_amounts: Some(vec![Decimal::new(123, 4), Decimal::new(456, 4)]),
        nullable_fees: vec![Some(Decimal::new(100, 2)), None, Some(Decimal::new(200, 2))],

        ts_us: ts,
        ts_ns: ts,
        opt_ts: Some(ts),
        ts_log: vec![ts, ts],
        nullable_ts_log: vec![Some(ts), None, Some(ts)],

        default_price: Decimal::new(99, 2),
        default_ts: ts,
    };

    // Schema reflects each per-field override exactly. A regression here means
    // the override either didn't reach the type-registry mapping or got
    // overwritten by the default.
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
        schema_dtype(&schema, "ts_us"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "ts_ns"),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "opt_ts"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        schema_dtype(&schema, "ts_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Nanoseconds, None)))
    );
    assert_eq!(
        schema_dtype(&schema, "nullable_ts_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Nanoseconds, None)))
    );
    assert_eq!(
        schema_dtype(&schema, "default_price"),
        DataType::Decimal(38, 10)
    );
    assert_eq!(
        schema_dtype(&schema, "default_ts"),
        DataType::Datetime(TimeUnit::Milliseconds, None)
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
        dtype(&df, "ts_us"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        dtype(&df, "ts_ns"),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    assert_eq!(
        dtype(&df, "opt_ts"),
        DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(
        dtype(&df, "ts_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Nanoseconds, None)))
    );
    assert_eq!(
        dtype(&df, "nullable_ts_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Nanoseconds, None)))
    );
    assert_eq!(dtype(&df, "default_price"), DataType::Decimal(38, 10));
    assert_eq!(
        dtype(&df, "default_ts"),
        DataType::Datetime(TimeUnit::Milliseconds, None)
    );

    // The columnar batch path threads through different codegen than the
    // single-row path (typed list builders, bulk emitters, etc.) — assert it
    // produces the same dtypes on a multi-row slice.
    let batch = vec![row.clone(), row.clone()];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.height(), 2);
    assert_eq!(dtype(&df_batch, "price"), DataType::Decimal(18, 6));
    assert_eq!(
        dtype(&df_batch, "ts_ns"),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    assert_eq!(
        dtype(&df_batch, "ts_log"),
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Nanoseconds, None)))
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
        DataType::List(Box::new(DataType::Datetime(TimeUnit::Nanoseconds, None)))
    );

    // The chrono call selected by `time_unit = "ns"` (`timestamp_nanos_opt`)
    // returns values 1_000_000x larger than `timestamp_millis`. A bug that
    // kept the buffer in millis but only changed the cast dtype would produce
    // the *same* underlying i64 here as for `default_ts`, so spot-check the
    // ns column actually scaled up.
    fn datetime_value(av: AnyValue<'_>, expected_unit: TimeUnit) -> i64 {
        match av {
            AnyValue::Datetime(v, u, _) | AnyValue::DatetimeOwned(v, u, _) => {
                assert_eq!(u, expected_unit, "wrong TimeUnit on AnyValue");
                v
            }
            other => panic!("unexpected Datetime AnyValue: {other:?}"),
        }
    }
    let ns_int = datetime_value(
        df.column("ts_ns").unwrap().get(0).unwrap(),
        TimeUnit::Nanoseconds,
    );
    let ms_int = datetime_value(
        df.column("default_ts").unwrap().get(0).unwrap(),
        TimeUnit::Milliseconds,
    );
    assert_eq!(ns_int, ms_int.checked_mul(1_000_000).unwrap());

    // Empty dataframe also reports the override dtypes.
    let empty = OverrideEverything::empty_dataframe().unwrap();
    assert_eq!(dtype(&empty, "ts_ns"), DataType::Datetime(TimeUnit::Nanoseconds, None));
    assert_eq!(dtype(&empty, "price"), DataType::Decimal(18, 6));
}
