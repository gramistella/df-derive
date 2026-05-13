use crate::core::dataframe::ToDataFrame;
use df_derive::ToDataFrame;
use polars::prelude::PolarsResult;
use std::fmt;

#[derive(Clone, Copy)]
struct FailingDisplay;

impl fmt::Display for FailingDisplay {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
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
