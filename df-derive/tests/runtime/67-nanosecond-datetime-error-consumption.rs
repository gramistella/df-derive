use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use df_derive::ToDataFrame;
use df_derive::dataframe::Columnar;
use std::cell::Cell;

#[derive(ToDataFrame)]
struct DateTimeNanosecondsRow {
    #[df_derive(time_unit = "ns")]
    values: Vec<DateTime<Utc>>,
}

#[derive(ToDataFrame)]
struct NaiveDateTimeNanosecondsRow {
    #[df_derive(time_unit = "ns")]
    values: Vec<NaiveDateTime>,
}

fn utc_year(year: i32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0).single().unwrap()
}

fn naive_year(year: i32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(year, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
}

#[test]
fn datetime_nanosecond_errors_do_not_consume_later_source_rows() {
    let out_of_range = utc_year(2500);
    assert!(out_of_range.timestamp_nanos_opt().is_none());

    let yielded = Cell::new(0);
    let rows = [
        DateTimeNanosecondsRow {
            values: vec![out_of_range],
        },
        DateTimeNanosecondsRow {
            values: vec![utc_year(1970)],
        },
    ];
    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));
    let error = DateTimeNanosecondsRow::encode(inspected).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("DateTime<Tz> value is out of range for nanosecond timestamps"),
        "unexpected error: {error}",
    );
    assert_eq!(yielded.get(), 1);
}

#[test]
fn naive_datetime_nanosecond_errors_do_not_consume_later_source_rows() {
    let out_of_range = naive_year(2500);
    assert!(out_of_range.and_utc().timestamp_nanos_opt().is_none(),);

    let yielded = Cell::new(0);
    let rows = [
        NaiveDateTimeNanosecondsRow {
            values: vec![out_of_range],
        },
        NaiveDateTimeNanosecondsRow {
            values: vec![naive_year(1970)],
        },
    ];
    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));
    let error = NaiveDateTimeNanosecondsRow::encode(inspected).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("NaiveDateTime value is out of range for nanosecond timestamps"),
        "unexpected error: {error}",
    );
    assert_eq!(yielded.get(), 1);
}
