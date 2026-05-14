#![deny(clippy::clone_on_copy, clippy::unwrap_used)]

use chrono::NaiveDate;
use df_derive::ToDataFrame;
use polars::prelude::*;

#[path = "common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame)]
struct StrictLintRow {
    i: i32,
    u: u64,
    b: bool,
    d: NaiveDate,
    opt_i: Option<i32>,
    opt_b: Option<bool>,
    opt_d: Option<NaiveDate>,
}

#[test]
fn generated_copy_and_date_code_is_strict_lint_friendly() -> PolarsResult<()> {
    let date = NaiveDate::from_ymd_opt(1970, 1, 2)
        .ok_or_else(|| polars_err!(ComputeError: "invalid fixed test date"))?;
    let row = StrictLintRow {
        i: 1,
        u: 2,
        b: true,
        d: date,
        opt_i: Some(3),
        opt_b: Some(false),
        opt_d: Some(date),
    };

    let single = row.to_dataframe()?;
    assert_eq!(single.column("i")?.get(0)?, AnyValue::Int32(1));
    assert_eq!(single.column("d")?.get(0)?, AnyValue::Date(1));

    let rows = [row];
    let batch = rows.as_slice().to_dataframe()?;
    assert_eq!(batch.column("u")?.get(0)?, AnyValue::UInt64(2));
    assert_eq!(batch.column("b")?.get(0)?, AnyValue::Boolean(true));
    assert_eq!(batch.column("opt_i")?.get(0)?, AnyValue::Int32(3));
    assert_eq!(batch.column("opt_b")?.get(0)?, AnyValue::Boolean(false));
    assert_eq!(batch.column("opt_d")?.get(0)?, AnyValue::Date(1));

    Ok(())
}
