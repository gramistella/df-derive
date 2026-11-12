//! Encoding rows two ways and requiring the same frame.
//!
//! Every builder defines its encoding one slot at a time, in `push`. Rows are
//! actually encoded a block at a time. The helpers here do both and compare.
//! Test files pull this module in with `#[path = "support/agreement.rs"]`; it
//! lives in a sub-directory so Cargo does not take it for a test of its own.

#![allow(dead_code)]

use df_derive::dataframe::{ColumnBuilder, Columnar, ToDataFrame, ToDataFrameVec};
use polars::prelude::*;

/// Encodes one slot at a time: the definition every builder gives in `push`.
pub fn one_at_a_time<T: Columnar>(rows: &[T]) -> DataFrame {
    let mut builder = <T::Builder as ColumnBuilder<T>>::new();
    for row in rows {
        builder.push(Some(row)).unwrap();
    }
    let mut columns = Vec::new();
    builder
        .finish(PlSmallStr::EMPTY, None, &mut columns)
        .unwrap();
    DataFrame::new(rows.len(), columns).unwrap()
}

/// The first row that cannot be pushed, if any.
pub fn first_rejected<T: Columnar>(rows: &[T]) -> Option<usize> {
    let mut builder = <T::Builder as ColumnBuilder<T>>::new();
    rows.iter().position(|row| builder.push(Some(row)).is_err())
}

#[track_caller]
pub fn assert_same(what: &str, actual: &DataFrame, expected: &DataFrame) {
    assert_eq!(actual.schema(), expected.schema(), "{what}: schema");
    assert_eq!(actual.height(), expected.height(), "{what}: height");
    for (actual, expected) in actual.columns().iter().zip(expected.columns()) {
        assert!(
            actual.equals_missing(expected),
            "{what}: column `{}` differs\n{actual:?}\n{expected:?}",
            actual.name()
        );
    }
}

/// Every way of encoding `rows` gives the frame that pushing them gives.
#[track_caller]
pub fn check<T: ToDataFrame>(rows: &[T]) {
    let expected = one_at_a_time(rows);
    let len = rows.len();

    let encoded = rows.to_dataframe().unwrap();
    assert_same(&format!("{len} rows in a slice"), &encoded, &expected);
    for column in encoded.columns() {
        assert_eq!(
            column.as_materialized_series().n_chunks(),
            1,
            "`{}`",
            column.name()
        );
    }

    let by_reference = T::rows_to_dataframe(rows.iter()).unwrap();
    assert_same(
        &format!("{len} rows by reference"),
        &by_reference,
        &expected,
    );

    // Where the rows are cut makes no difference to what they encode to.
    for cut in [len / 3, len / 2] {
        let mut stacked = rows[..cut].to_dataframe().unwrap();
        stacked
            .vstack_mut(&rows[cut..].to_dataframe().unwrap())
            .unwrap();
        assert_same(&format!("{len} rows cut at {cut}"), &stacked, &expected);
    }
}
