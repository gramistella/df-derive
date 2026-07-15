use df_derive::ToDataFrame;

#[path = "../support/local_runtime.rs"]
mod runtime_support;

mod custom_runtime {
    pub use super::runtime_support::dataframe::{
        ColumnSink, Columnar, ColumnarSpec, RowCursor, ToDataFrame,
    };
}

#[derive(ToDataFrame)]
#[allow(clippy::type_complexity)]
#[df_derive(
    trait = "custom_runtime::ToDataFrame",
    columnar = "custom_runtime::Columnar"
)]
struct CustomRuntimeReplayRow {
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

fn replay_row(base: i64) -> CustomRuntimeReplayRow {
    CustomRuntimeReplayRow {
        values: (
            base,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 5,
            base + 6,
            base + 7,
            base + 8,
            base + 9,
            base + 10,
            base + 11,
            base + 12,
            base + 13,
            base + 14,
            base + 15,
        ),
    }
}

#[test]
fn custom_runtime_replays_general_iterators() {
    fn assert_custom_runtime<T: custom_runtime::ToDataFrame>() {}

    assert_custom_runtime::<CustomRuntimeReplayRow>();
    const {
        assert!(
            <CustomRuntimeReplayRow as custom_runtime::ColumnarSpec>::REQUIRES_ROW_REPLAY,
            "the wide tuple must select the custom runtime's replay cursor",
        );
    }

    let rows = [replay_row(0), replay_row(100), replay_row(200)];
    let general_iterator = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| (index != 1).then_some(row));
    let frame = <CustomRuntimeReplayRow as custom_runtime::Columnar>::encode(general_iterator)
        .expect("custom runtime should encode replayed rows");

    assert_eq!(frame.shape(), (2, 16));
    for index in 0..16 {
        let values: Vec<i64> = frame
            .column(&format!("values.field_{index}"))
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, [index, 200 + index].map(i64::from));
    }
}
