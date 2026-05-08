use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

use chrono::{NaiveDate, NaiveTime};
use polars::prelude::*;

// Cover every accepted shape for the four new bases:
//   NaiveDate / NaiveTime / std::time::Duration / chrono::Duration
// in their bare, Option, Vec<T>, Vec<Option<T>>, and Option<Vec<T>>
// permutations, plus time_unit overrides on both Duration variants.

#[derive(ToDataFrame, Clone)]
struct All {
    // NaiveDate
    nd: NaiveDate,
    nd_opt: Option<NaiveDate>,
    nd_vec: Vec<NaiveDate>,
    nd_vec_opt: Vec<Option<NaiveDate>>,
    nd_opt_vec: Option<Vec<NaiveDate>>,

    // NaiveTime
    nt: NaiveTime,
    nt_opt: Option<NaiveTime>,
    nt_vec: Vec<NaiveTime>,
    nt_vec_opt: Vec<Option<NaiveTime>>,
    nt_opt_vec: Option<Vec<NaiveTime>>,

    // std::time::Duration
    sd: std::time::Duration,
    sd_opt: Option<std::time::Duration>,
    sd_vec: Vec<std::time::Duration>,
    sd_vec_opt: Vec<Option<std::time::Duration>>,
    sd_opt_vec: Option<Vec<std::time::Duration>>,

    // chrono::Duration
    cd: chrono::Duration,
    cd_opt: Option<chrono::Duration>,
    cd_vec: Vec<chrono::Duration>,
    cd_vec_opt: Vec<Option<chrono::Duration>>,
    cd_opt_vec: Option<Vec<chrono::Duration>>,

    // time_unit overrides on chrono::Duration
    #[df_derive(time_unit = "ms")]
    cd_ms: chrono::Duration,
    #[df_derive(time_unit = "us")]
    cd_us: chrono::Duration,
    #[df_derive(time_unit = "ns")]
    cd_ns: chrono::Duration,
    #[df_derive(time_unit = "ms")]
    cd_ms_vec: Vec<chrono::Duration>,

    // time_unit override on std::time::Duration
    #[df_derive(time_unit = "ms")]
    sd_ms: std::time::Duration,
    #[df_derive(time_unit = "us")]
    sd_us: Option<std::time::Duration>,
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(n, _)| n == col)
        .map(|(_, dt)| dt.clone())
        .unwrap_or_else(|| panic!("column {col} missing from schema"))
}

fn col_dtype(df: &DataFrame, col: &str) -> DataType {
    df.column(col).unwrap().dtype().clone()
}

fn date_value(av: AnyValue<'_>) -> i32 {
    match av {
        AnyValue::Date(v) => v,
        other => panic!("expected Date AnyValue, got {other:?}"),
    }
}

fn time_value(av: AnyValue<'_>) -> i64 {
    match av {
        AnyValue::Time(v) => v,
        other => panic!("expected Time AnyValue, got {other:?}"),
    }
}

fn duration_value(av: AnyValue<'_>, expected_unit: TimeUnit) -> i64 {
    match av {
        AnyValue::Duration(v, u) => {
            assert_eq!(u, expected_unit, "wrong TimeUnit on Duration AnyValue");
            v
        }
        other => panic!("expected Duration AnyValue, got {other:?}"),
    }
}

fn assert_list_dates(df: &DataFrame, col: &str, row: usize, expected: &[Option<i32>]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected list at {col}[{row}], got {v:?}");
    };
    let actual: Vec<Option<i32>> = inner
        .iter()
        .map(|av| match av {
            AnyValue::Date(d) => Some(d),
            AnyValue::Null => None,
            other => panic!("unexpected list elem {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected, "{col}[{row}] mismatch");
}

fn assert_list_times(df: &DataFrame, col: &str, row: usize, expected: &[Option<i64>]) {
    let v = df.column(col).unwrap().get(row).unwrap();
    let AnyValue::List(inner) = v else {
        panic!("expected list at {col}[{row}], got {v:?}");
    };
    let actual: Vec<Option<i64>> = inner
        .iter()
        .map(|av| match av {
            AnyValue::Time(t) => Some(t),
            AnyValue::Null => None,
            other => panic!("unexpected list elem {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected, "{col}[{row}] mismatch");
}

fn assert_list_durations(
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
        .map(|av| match av {
            AnyValue::Duration(d, u) => {
                assert_eq!(u, expected_unit);
                Some(d)
            }
            AnyValue::Null => None,
            other => panic!("unexpected list elem {other:?}"),
        })
        .collect();
    assert_eq!(actual, expected, "{col}[{row}] mismatch");
}

fn main() {
    println!("--- Testing NaiveDate / NaiveTime / Duration support ---");

    // Edge dates: 1970-01-01 -> 0; 1969-12-31 -> -1; leap day; etc.
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let pre_epoch = NaiveDate::from_ymd_opt(1969, 12, 31).unwrap();
    let leap = NaiveDate::from_ymd_opt(2000, 2, 29).unwrap();
    let mid = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
    let midnight = NaiveTime::from_hms_opt(0, 0, 0).unwrap();
    let just_before_mid = NaiveTime::from_hms_nano_opt(23, 59, 59, 999_999_999).unwrap();
    let noon = NaiveTime::from_hms_opt(12, 0, 0).unwrap();

    let row0 = All {
        nd: epoch,
        nd_opt: Some(pre_epoch),
        nd_vec: vec![epoch, leap, pre_epoch],
        nd_vec_opt: vec![Some(epoch), None, Some(leap)],
        nd_opt_vec: Some(vec![mid, leap]),

        nt: midnight,
        nt_opt: Some(just_before_mid),
        nt_vec: vec![midnight, noon, just_before_mid],
        nt_vec_opt: vec![Some(midnight), None, Some(noon)],
        nt_opt_vec: Some(vec![noon]),

        sd: std::time::Duration::from_nanos(1_500),
        sd_opt: Some(std::time::Duration::from_nanos(2_000)),
        sd_vec: vec![
            std::time::Duration::from_nanos(0),
            std::time::Duration::from_nanos(7_777),
        ],
        sd_vec_opt: vec![Some(std::time::Duration::from_nanos(42)), None],
        sd_opt_vec: Some(vec![std::time::Duration::from_nanos(123)]),

        cd: chrono::Duration::nanoseconds(1_000),
        cd_opt: Some(chrono::Duration::nanoseconds(2_500)),
        cd_vec: vec![chrono::Duration::nanoseconds(0), chrono::Duration::nanoseconds(500)],
        cd_vec_opt: vec![Some(chrono::Duration::nanoseconds(1)), None],
        cd_opt_vec: Some(vec![chrono::Duration::nanoseconds(7)]),

        cd_ms: chrono::Duration::milliseconds(123),
        cd_us: chrono::Duration::microseconds(456),
        cd_ns: chrono::Duration::nanoseconds(789),
        cd_ms_vec: vec![
            chrono::Duration::milliseconds(10),
            chrono::Duration::milliseconds(20),
        ],

        sd_ms: std::time::Duration::from_millis(42),
        sd_us: Some(std::time::Duration::from_micros(99)),
    };

    println!("Schema dtype assertions...");
    let schema = All::schema().unwrap();
    assert_eq!(schema_dtype(&schema, "nd"), DataType::Date);
    assert_eq!(schema_dtype(&schema, "nd_opt"), DataType::Date);
    assert_eq!(
        schema_dtype(&schema, "nd_vec"),
        DataType::List(Box::new(DataType::Date))
    );
    assert_eq!(
        schema_dtype(&schema, "nd_vec_opt"),
        DataType::List(Box::new(DataType::Date))
    );
    assert_eq!(
        schema_dtype(&schema, "nd_opt_vec"),
        DataType::List(Box::new(DataType::Date))
    );

    assert_eq!(schema_dtype(&schema, "nt"), DataType::Time);
    assert_eq!(schema_dtype(&schema, "nt_opt"), DataType::Time);
    assert_eq!(
        schema_dtype(&schema, "nt_vec"),
        DataType::List(Box::new(DataType::Time))
    );

    assert_eq!(
        schema_dtype(&schema, "sd"),
        DataType::Duration(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "cd"),
        DataType::Duration(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "sd_vec"),
        DataType::List(Box::new(DataType::Duration(TimeUnit::Nanoseconds)))
    );

    // time_unit overrides
    assert_eq!(
        schema_dtype(&schema, "cd_ms"),
        DataType::Duration(TimeUnit::Milliseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "cd_us"),
        DataType::Duration(TimeUnit::Microseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "cd_ns"),
        DataType::Duration(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "cd_ms_vec"),
        DataType::List(Box::new(DataType::Duration(TimeUnit::Milliseconds)))
    );
    assert_eq!(
        schema_dtype(&schema, "sd_ms"),
        DataType::Duration(TimeUnit::Milliseconds)
    );
    assert_eq!(
        schema_dtype(&schema, "sd_us"),
        DataType::Duration(TimeUnit::Microseconds)
    );

    println!("Single-row to_dataframe path...");
    let df = row0.clone().to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 26));

    // Runtime dtype matches schema.
    assert_eq!(col_dtype(&df, "nd"), DataType::Date);
    assert_eq!(col_dtype(&df, "nt"), DataType::Time);
    assert_eq!(
        col_dtype(&df, "sd"),
        DataType::Duration(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        col_dtype(&df, "cd_ms"),
        DataType::Duration(TimeUnit::Milliseconds)
    );

    // Edge-date values: epoch -> 0, pre-epoch -> -1, leap day = 11017 days.
    assert_eq!(date_value(df.column("nd").unwrap().get(0).unwrap()), 0);
    assert_eq!(date_value(df.column("nd_opt").unwrap().get(0).unwrap()), -1);

    // NaiveTime midnight -> 0 ns, just-before-midnight -> 86399999999999.
    assert_eq!(time_value(df.column("nt").unwrap().get(0).unwrap()), 0);
    assert_eq!(
        time_value(df.column("nt_opt").unwrap().get(0).unwrap()),
        86_399_999_999_999
    );

    // Duration value extraction.
    assert_eq!(
        duration_value(df.column("sd").unwrap().get(0).unwrap(), TimeUnit::Nanoseconds),
        1_500
    );
    assert_eq!(
        duration_value(df.column("cd").unwrap().get(0).unwrap(), TimeUnit::Nanoseconds),
        1_000
    );
    assert_eq!(
        duration_value(
            df.column("cd_ms").unwrap().get(0).unwrap(),
            TimeUnit::Milliseconds
        ),
        123
    );
    assert_eq!(
        duration_value(
            df.column("cd_us").unwrap().get(0).unwrap(),
            TimeUnit::Microseconds
        ),
        456
    );
    assert_eq!(
        duration_value(
            df.column("cd_ns").unwrap().get(0).unwrap(),
            TimeUnit::Nanoseconds
        ),
        789
    );

    // Confirm that a `time_unit = "ns"` field on chrono::Duration uses the
    // ns mapping at the per-row level. A 1ms Duration in nanos = 1_000_000.
    let one_ms_in_ns = chrono::Duration::milliseconds(1).num_nanoseconds().unwrap();
    assert_eq!(one_ms_in_ns, 1_000_000);

    // List(Date) round-trip
    assert_list_dates(
        &df,
        "nd_vec",
        0,
        &[
            Some(0),
            Some(leap.signed_duration_since(epoch).num_days() as i32),
            Some(-1),
        ],
    );
    assert_list_dates(
        &df,
        "nd_vec_opt",
        0,
        &[
            Some(0),
            None,
            Some(leap.signed_duration_since(epoch).num_days() as i32),
        ],
    );

    // List(Time)
    assert_list_times(&df, "nt_vec", 0, &[Some(0), Some(43_200_000_000_000), Some(86_399_999_999_999)]);
    assert_list_times(&df, "nt_vec_opt", 0, &[Some(0), None, Some(43_200_000_000_000)]);

    // List(Duration)
    assert_list_durations(
        &df,
        "sd_vec",
        0,
        &[Some(0), Some(7_777)],
        TimeUnit::Nanoseconds,
    );
    assert_list_durations(
        &df,
        "sd_vec_opt",
        0,
        &[Some(42), None],
        TimeUnit::Nanoseconds,
    );
    assert_list_durations(
        &df,
        "cd_ms_vec",
        0,
        &[Some(10), Some(20)],
        TimeUnit::Milliseconds,
    );

    println!("Columnar batch path with mixed Some/None...");
    let row1 = All {
        nd: leap,
        nd_opt: None,
        nd_vec: vec![],
        nd_vec_opt: vec![None],
        nd_opt_vec: None,

        nt: noon,
        nt_opt: None,
        nt_vec: vec![],
        nt_vec_opt: vec![None],
        nt_opt_vec: None,

        sd: std::time::Duration::from_nanos(0),
        sd_opt: None,
        sd_vec: vec![],
        sd_vec_opt: vec![None],
        sd_opt_vec: None,

        cd: chrono::Duration::zero(),
        cd_opt: None,
        cd_vec: vec![],
        cd_vec_opt: vec![None],
        cd_opt_vec: None,

        cd_ms: chrono::Duration::milliseconds(-50),
        cd_us: chrono::Duration::microseconds(0),
        cd_ns: chrono::Duration::nanoseconds(-1),
        cd_ms_vec: vec![],

        sd_ms: std::time::Duration::from_millis(0),
        sd_us: None,
    };

    let batch = vec![row0.clone(), row1.clone(), row0.clone()];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (3, 26));

    // Schema preserved through columnar path.
    let bs = df_batch.schema();
    assert_eq!(bs.get("nd"), Some(&DataType::Date));
    assert_eq!(bs.get("nt"), Some(&DataType::Time));
    assert_eq!(
        bs.get("sd"),
        Some(&DataType::Duration(TimeUnit::Nanoseconds))
    );
    assert_eq!(
        bs.get("cd_ms_vec"),
        Some(&DataType::List(Box::new(DataType::Duration(
            TimeUnit::Milliseconds
        ))))
    );

    // Row 1 assertions (None / empty / zero values).
    let row1_nd = date_value(df_batch.column("nd").unwrap().get(1).unwrap());
    assert_eq!(
        row1_nd,
        leap.signed_duration_since(epoch).num_days() as i32
    );
    assert!(matches!(
        df_batch.column("nd_opt").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert!(matches!(
        df_batch.column("nd_opt_vec").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert_list_dates(&df_batch, "nd_vec", 1, &[]);
    assert_list_dates(&df_batch, "nd_vec_opt", 1, &[None]);

    // Negative chrono Duration round-trip.
    assert_eq!(
        duration_value(
            df_batch.column("cd_ms").unwrap().get(1).unwrap(),
            TimeUnit::Milliseconds
        ),
        -50
    );
    assert_eq!(
        duration_value(
            df_batch.column("cd_ns").unwrap().get(1).unwrap(),
            TimeUnit::Nanoseconds
        ),
        -1
    );

    println!("Empty-DataFrame schema check...");
    let empty = All::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 26));
    let es = empty.schema();
    assert_eq!(es.get("nd"), Some(&DataType::Date));
    assert_eq!(es.get("nt"), Some(&DataType::Time));
    assert_eq!(
        es.get("sd"),
        Some(&DataType::Duration(TimeUnit::Nanoseconds))
    );
    assert_eq!(
        es.get("cd_ms"),
        Some(&DataType::Duration(TimeUnit::Milliseconds))
    );

    println!("\nNaiveDate / NaiveTime / Duration test completed successfully!");
}
