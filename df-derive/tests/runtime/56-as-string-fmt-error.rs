use df_derive::ToDataFrame;
use df_derive::dataframe::{Columnar, ToDataFrame};
use polars::prelude::PolarsResult;
use std::{cell::Cell, fmt};

#[derive(Clone, Copy)]
struct FailingDisplay;

impl fmt::Display for FailingDisplay {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}

#[derive(Clone, Copy)]
enum ConditionalDisplay {
    Value(i32),
    Fail,
}

impl fmt::Display for ConditionalDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(value) => value.fmt(f),
            Self::Fail => Err(fmt::Error),
        }
    }
}

#[derive(ToDataFrame)]
struct ScalarAsString {
    #[df_derive(as_string)]
    value: FailingDisplay,
}

#[derive(ToDataFrame)]
struct OptionalAsString {
    #[df_derive(as_string)]
    value: Option<FailingDisplay>,
}

#[derive(ToDataFrame)]
struct VecAsString {
    #[df_derive(as_string)]
    values: Vec<FailingDisplay>,
}

#[derive(ToDataFrame)]
struct VecOptionalAsString {
    #[df_derive(as_string)]
    values: Vec<Option<FailingDisplay>>,
}

#[derive(ToDataFrame)]
struct VecConditionalAsString {
    #[df_derive(as_string)]
    values: Vec<ConditionalDisplay>,
}

fn assert_display_error<T>(result: PolarsResult<T>) {
    let Err(err) = result else {
        panic!("expected as_string Display formatting error");
    };
    let message = err.to_string();
    assert!(
        message.contains("df-derive: as_string Display formatting failed"),
        "unexpected error: {message}",
    );
}

#[test]
fn as_string_display_errors_are_polars_errors() {
    assert_display_error(
        ScalarAsString {
            value: FailingDisplay,
        }
        .to_dataframe(),
    );
    assert_display_error(
        OptionalAsString {
            value: Some(FailingDisplay),
        }
        .to_dataframe(),
    );
    assert_display_error(
        VecAsString {
            values: vec![FailingDisplay],
        }
        .to_dataframe(),
    );
    assert_display_error(
        VecOptionalAsString {
            values: vec![Some(FailingDisplay)],
        }
        .to_dataframe(),
    );
}

#[test]
fn list_display_errors_do_not_consume_later_source_rows() {
    let yielded = Cell::new(0);
    let rows = [
        VecAsString {
            values: vec![FailingDisplay],
        },
        VecAsString {
            values: vec![FailingDisplay],
        },
    ];
    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));

    assert_display_error(VecAsString::encode(inspected));
    assert_eq!(yielded.get(), 1);
}

#[test]
fn list_display_errors_preserve_a_successful_source_prefix() {
    let yielded = Cell::new(0);
    let rows = [
        VecConditionalAsString {
            values: vec![ConditionalDisplay::Value(1), ConditionalDisplay::Value(2)],
        },
        VecConditionalAsString {
            values: vec![ConditionalDisplay::Value(3), ConditionalDisplay::Fail],
        },
        VecConditionalAsString {
            values: vec![ConditionalDisplay::Value(4)],
        },
    ];
    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));

    assert_display_error(VecConditionalAsString::encode(inspected));
    assert_eq!(yielded.get(), 2);
}
