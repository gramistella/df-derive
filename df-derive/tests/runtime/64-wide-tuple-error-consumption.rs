use df_derive::ToDataFrame;
use df_derive::dataframe::{Columnar, ToDataFrameVec};
use rust_decimal::Decimal;
use std::cell::Cell;
use std::fmt;
use std::rc::Rc;

#[derive(ToDataFrame)]
#[allow(clippy::type_complexity)]
struct WideFallibleTuple {
    values: (
        Decimal,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ),
}

#[derive(ToDataFrame)]
struct NestedReplayingChild {
    child: WideFallibleTuple,
}

#[derive(Clone, Copy)]
struct FailingNestedDisplay;

impl fmt::Display for FailingNestedDisplay {
    fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}

#[derive(Clone)]
struct CountingNestedDisplay(Rc<Cell<usize>>);

impl fmt::Display for CountingNestedDisplay {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.set(self.0.get() + 1);
        formatter.write_str("later")
    }
}

#[derive(ToDataFrame)]
struct FallibleNestedChild {
    #[df_derive(as_string)]
    failing: FailingNestedDisplay,
    #[df_derive(as_string)]
    later: CountingNestedDisplay,
}

#[derive(ToDataFrame)]
struct ParentWithFallibleNestedChild {
    child: FallibleNestedChild,
}

fn wide_tuple(decimal: Decimal) -> WideFallibleTuple {
    WideFallibleTuple {
        values: (
            decimal, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ),
    }
}

#[test]
fn wide_tuple_decimal_errors_do_not_consume_later_source_rows() {
    let yielded = Cell::new(0);
    let rows = [
        wide_tuple(Decimal::new(123, 10)),
        wide_tuple(Decimal::MAX),
        wide_tuple(Decimal::new(456, 10)),
    ];
    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));

    let error = WideFallibleTuple::encode(inspected).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("decimal mantissa rescale to scale 10 failed"),
        "unexpected error: {error}",
    );
    assert_eq!(yielded.get(), 2);
}

#[test]
fn mixed_wide_tuple_preserves_terminal_order_after_hybrid_replay() {
    let rows = [
        wide_tuple(Decimal::new(123, 2)),
        wide_tuple(Decimal::new(456, 2)),
    ];
    let frame = WideFallibleTuple::encode(rows.iter()).unwrap();

    assert_eq!(frame.shape(), (2, 17));
    let expected_names: Vec<String> = (0..17)
        .map(|index| format!("values.field_{index}"))
        .collect();
    let actual_names: Vec<String> = frame
        .get_column_names()
        .into_iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(actual_names, expected_names);
    for index in 1..17 {
        assert_eq!(
            frame
                .column(&format!("values.field_{index}"))
                .unwrap()
                .i64()
                .unwrap()
                .get(1),
            Some(i64::from(index - 1)),
        );
    }
}

#[test]
fn nested_replaying_child_preserves_rows_and_terminal_order() {
    let rows = [
        NestedReplayingChild {
            child: wide_tuple(Decimal::new(123, 2)),
        },
        NestedReplayingChild {
            child: wide_tuple(Decimal::new(456, 2)),
        },
    ];
    let frame = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(frame.shape(), (2, 17));
    let expected_names: Vec<String> = (0..17)
        .map(|index| format!("child.values.field_{index}"))
        .collect();
    let actual_names: Vec<String> = frame
        .get_column_names()
        .into_iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(actual_names, expected_names);
    for index in 1..17 {
        assert_eq!(
            frame
                .column(&format!("child.values.field_{index}"))
                .unwrap()
                .i64()
                .unwrap()
                .get(1),
            Some(i64::from(index - 1)),
        );
    }
}

#[test]
fn nested_child_conversion_errors_skip_later_child_conversions() {
    let later_calls = Rc::new(Cell::new(0));
    let rows = [
        ParentWithFallibleNestedChild {
            child: FallibleNestedChild {
                failing: FailingNestedDisplay,
                later: CountingNestedDisplay(Rc::clone(&later_calls)),
            },
        },
        ParentWithFallibleNestedChild {
            child: FallibleNestedChild {
                failing: FailingNestedDisplay,
                later: CountingNestedDisplay(Rc::clone(&later_calls)),
            },
        },
    ];

    let error = ParentWithFallibleNestedChild::encode(rows.iter()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("df-derive: as_string Display formatting failed"),
        "unexpected error: {error}",
    );
    assert_eq!(
        later_calls.get(),
        0,
        "columns after the failing nested conversion must not be evaluated",
    );
}
