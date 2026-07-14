use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
#[allow(clippy::type_complexity)]
pub struct TupleReplayBoundaryMinusOne {
    scalar: i64,
    values: (
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
#[allow(clippy::type_complexity)]
pub struct TupleReplayBoundary {
    values: (
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

fn row_value(row: usize, offset: i64) -> i64 {
    i64::try_from(row).unwrap() * 16 + offset
}

pub fn make_tuple_replay_boundary_minus_one(row_count: usize) -> Vec<TupleReplayBoundaryMinusOne> {
    (0..row_count)
        .map(|row| TupleReplayBoundaryMinusOne {
            scalar: row_value(row, 0),
            values: (
                row_value(row, 1),
                row_value(row, 2),
                row_value(row, 3),
                row_value(row, 4),
                row_value(row, 5),
                row_value(row, 6),
                row_value(row, 7),
                row_value(row, 8),
                row_value(row, 9),
                row_value(row, 10),
                row_value(row, 11),
                row_value(row, 12),
                row_value(row, 13),
                row_value(row, 14),
                row_value(row, 15),
            ),
        })
        .collect()
}

pub fn make_tuple_replay_boundary(row_count: usize) -> Vec<TupleReplayBoundary> {
    (0..row_count)
        .map(|row| TupleReplayBoundary {
            values: (
                row_value(row, 0),
                row_value(row, 1),
                row_value(row, 2),
                row_value(row, 3),
                row_value(row, 4),
                row_value(row, 5),
                row_value(row, 6),
                row_value(row, 7),
                row_value(row, 8),
                row_value(row, 9),
                row_value(row, 10),
                row_value(row, 11),
                row_value(row, 12),
                row_value(row, 13),
                row_value(row, 14),
                row_value(row, 15),
            ),
        })
        .collect()
}
