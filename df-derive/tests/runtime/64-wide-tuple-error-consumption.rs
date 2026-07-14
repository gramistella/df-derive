use df_derive::ToDataFrame;
use df_derive::dataframe::Columnar;
use rust_decimal::Decimal;
use std::cell::Cell;

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
    let rows = [wide_tuple(Decimal::MAX), wide_tuple(Decimal::MAX)];
    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));

    let error = WideFallibleTuple::encode(inspected).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("decimal mantissa rescale to scale 10 failed"),
        "unexpected error: {error}",
    );
    assert_eq!(yielded.get(), 1);
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
