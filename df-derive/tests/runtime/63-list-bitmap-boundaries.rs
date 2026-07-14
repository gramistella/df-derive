use df_derive::ToDataFrame;
use df_derive::dataframe::{Columnar, ToDataFrameVec};
use polars::prelude::{AnyValue, DataType};
use rust_decimal::Decimal;
use std::cell::Cell;

const BOUNDARY_LENGTHS: [usize; 6] = [0, 1, 63, 64, 65, 129];

#[derive(Clone, Copy, Debug)]
enum ValidityPattern {
    AllValid,
    AllNull,
    WordBoundaryNulls,
}

impl ValidityPattern {
    const ALL: [Self; 3] = [Self::AllValid, Self::AllNull, Self::WordBoundaryNulls];

    const fn is_null(self, index: usize) -> bool {
        match self {
            Self::AllValid => false,
            Self::AllNull => true,
            // Exercise the last bit before a boundary, both sides of the
            // 64-bit boundary, and both sides of the second boundary.
            Self::WordBoundaryNulls => matches!(index, 0 | 62 | 63 | 64 | 127 | 128),
        }
    }
}

#[derive(ToDataFrame)]
struct BoolRow {
    values: Vec<bool>,
}

#[derive(ToDataFrame)]
struct OptionalBoolRow {
    values: Vec<Option<bool>>,
}

#[derive(ToDataFrame)]
struct NestedBoolRow {
    values: Vec<Vec<bool>>,
}

#[derive(ToDataFrame)]
struct NestedOptionalI32Row {
    values: Vec<Vec<Option<i32>>>,
}

#[derive(ToDataFrame)]
struct DeepOptionalBoolRow {
    values: Option<Vec<Option<Vec<Option<bool>>>>>,
}

#[derive(ToDataFrame)]
struct OptionalDecimalRow {
    #[df_derive(decimal(precision = 18, scale = 6))]
    values: Vec<Option<Decimal>>,
}

#[derive(ToDataFrame)]
#[allow(clippy::vec_box)]
struct NestedBoxBoolRow {
    values: Vec<Vec<Box<bool>>>,
}

fn bool_values(len: usize, pattern: ValidityPattern) -> Vec<Option<bool>> {
    (0..len)
        .map(|index| (!pattern.is_null(index)).then_some(index % 2 == 0))
        .collect()
}

#[test]
fn boolean_value_bitmaps_preserve_word_boundaries() {
    for len in BOUNDARY_LENGTHS {
        let expected: Vec<bool> = (0..len).map(|index| index % 2 == 0).collect();
        let rows = [BoolRow {
            values: expected.clone(),
        }];
        let dataframe = rows.as_slice().to_dataframe().unwrap();

        assert_eq!(dataframe.shape(), (1, 1), "len={len}");
        let value = dataframe.column("values").unwrap().get(0).unwrap();
        let AnyValue::List(inner) = value else {
            panic!("expected boolean list for len={len}, got {value:?}");
        };
        let actual: Vec<Option<bool>> = inner.bool().unwrap().iter().collect();
        let expected: Vec<Option<bool>> = expected.into_iter().map(Some).collect();

        assert_eq!(actual, expected, "len={len}");
    }
}

#[test]
fn nested_bare_boolean_segments_preserve_values_and_offsets() {
    let bools = |len| (0..len).map(|index| index % 3 == 0).collect::<Vec<_>>();
    let expected = [
        vec![],
        vec![vec![], bools(63)],
        vec![bools(64), vec![], bools(65)],
    ];
    let rows: Vec<NestedBoolRow> = expected
        .iter()
        .cloned()
        .map(|values| NestedBoolRow { values })
        .collect();
    let dataframe = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(dataframe.shape(), (expected.len(), 1));
    assert_eq!(
        dataframe.schema().get("values"),
        Some(&DataType::List(Box::new(DataType::List(Box::new(
            DataType::Boolean,
        ))))),
    );
    for (row_index, expected_outer) in expected.iter().enumerate() {
        let AnyValue::List(actual_outer) =
            dataframe.column("values").unwrap().get(row_index).unwrap()
        else {
            panic!("expected nested boolean list at row {row_index}");
        };
        assert_eq!(actual_outer.len(), expected_outer.len(), "row={row_index}");

        for (segment_index, expected_inner) in expected_outer.iter().enumerate() {
            let AnyValue::List(actual_inner) = actual_outer.get(segment_index).unwrap() else {
                panic!("expected boolean list at row {row_index}, segment {segment_index}");
            };
            let actual: Vec<Option<bool>> = actual_inner.bool().unwrap().iter().collect();
            assert_eq!(
                actual,
                expected_inner.iter().copied().map(Some).collect::<Vec<_>>(),
                "row={row_index} segment={segment_index}",
            );
        }
    }
}

#[test]
fn nested_boolean_segment_fast_path_respects_leaf_access_chains() {
    let expected = [vec![true, false, true], vec![], vec![false, false, true]];
    let rows = [NestedBoxBoolRow {
        values: expected
            .iter()
            .map(|segment| segment.iter().copied().map(Box::new).collect())
            .collect(),
    }];
    let dataframe = rows.as_slice().to_dataframe().unwrap();
    let AnyValue::List(outer) = dataframe.column("values").unwrap().get(0).unwrap() else {
        panic!("expected nested boolean list");
    };

    for (index, expected) in expected.iter().enumerate() {
        let AnyValue::List(inner) = outer.get(index).unwrap() else {
            panic!("expected inner boolean list at index {index}");
        };
        let actual: Vec<Option<bool>> = inner.bool().unwrap().iter().collect();
        assert_eq!(
            actual,
            expected.iter().copied().map(Some).collect::<Vec<_>>()
        );
    }
}

fn i32_values(len: usize, pattern: ValidityPattern) -> Vec<Option<i32>> {
    (0..len)
        .map(|index| {
            (!pattern.is_null(index)).then(|| i32::try_from(index).expect("test length fits i32"))
        })
        .collect()
}

fn segment_lengths(total: usize) -> Vec<usize> {
    if total == 0 {
        return vec![0];
    }

    let mut previous = 0;
    let mut lengths = Vec::new();
    for end in [1, 63, 64, 65] {
        if end <= total {
            lengths.push(end - previous);
            previous = end;
        }
    }
    if previous < total {
        lengths.push(total - previous);
    }
    lengths
}

fn segment_values(values: &[Option<i32>]) -> Vec<Vec<Option<i32>>> {
    let mut start = 0;
    segment_lengths(values.len())
        .into_iter()
        .map(|len| {
            let end = start + len;
            let segment = values[start..end].to_vec();
            start = end;
            segment
        })
        .collect()
}

#[test]
fn optional_boolean_bitmaps_preserve_word_boundaries() {
    for len in BOUNDARY_LENGTHS {
        for pattern in ValidityPattern::ALL {
            let expected = bool_values(len, pattern);
            let rows = [OptionalBoolRow {
                values: expected.clone(),
            }];
            let dataframe = rows.as_slice().to_dataframe().unwrap();

            assert_eq!(dataframe.shape(), (1, 1), "len={len} pattern={pattern:?}");
            assert_eq!(
                dataframe.schema().get("values"),
                Some(&DataType::List(Box::new(DataType::Boolean))),
                "len={len} pattern={pattern:?}",
            );
            let value = dataframe.column("values").unwrap().get(0).unwrap();
            let AnyValue::List(inner) = value else {
                panic!("expected boolean list for len={len} pattern={pattern:?}, got {value:?}");
            };
            let inner = inner.bool().unwrap();
            let actual: Vec<Option<bool>> = inner.iter().collect();

            assert_eq!(actual, expected, "len={len} pattern={pattern:?}");
            assert_eq!(
                inner.null_count(),
                expected.iter().filter(|value| value.is_none()).count(),
                "len={len} pattern={pattern:?}",
            );
        }
    }
}

#[test]
fn nested_optional_i32_bitmaps_preserve_segment_and_word_boundaries() {
    for len in BOUNDARY_LENGTHS {
        for pattern in ValidityPattern::ALL {
            let flat = i32_values(len, pattern);
            let expected = segment_values(&flat);
            let rows = [NestedOptionalI32Row {
                values: expected.clone(),
            }];
            let dataframe = rows.as_slice().to_dataframe().unwrap();

            assert_eq!(dataframe.shape(), (1, 1), "len={len} pattern={pattern:?}");
            assert_eq!(
                dataframe.schema().get("values"),
                Some(&DataType::List(Box::new(DataType::List(Box::new(
                    DataType::Int32,
                ))))),
                "len={len} pattern={pattern:?}",
            );
            let value = dataframe.column("values").unwrap().get(0).unwrap();
            let AnyValue::List(outer) = value else {
                panic!("expected nested list for len={len} pattern={pattern:?}, got {value:?}");
            };

            assert_eq!(outer.len(), expected.len(), "len={len} pattern={pattern:?}");
            for (segment_index, expected_segment) in expected.iter().enumerate() {
                let value = outer.get(segment_index).unwrap();
                let AnyValue::List(inner) = value else {
                    panic!(
                        "expected inner list at segment {segment_index} for len={len} \
                         pattern={pattern:?}, got {value:?}"
                    );
                };
                let inner = inner.i32().unwrap();
                let actual: Vec<Option<i32>> = inner.iter().collect();
                assert_eq!(
                    actual, *expected_segment,
                    "segment={segment_index} len={len} pattern={pattern:?}",
                );
                assert_eq!(
                    inner.null_count(),
                    expected_segment
                        .iter()
                        .filter(|value| value.is_none())
                        .count(),
                    "segment={segment_index} len={len} pattern={pattern:?}",
                );
            }
        }
    }
}

#[test]
fn deep_optional_boolean_shape_preserves_null_and_bitmap_boundaries() {
    let expected_segments = vec![
        None,
        Some(vec![]),
        Some(bool_values(63, ValidityPattern::WordBoundaryNulls)),
        Some(bool_values(64, ValidityPattern::AllNull)),
        Some(bool_values(65, ValidityPattern::WordBoundaryNulls)),
        Some(bool_values(129, ValidityPattern::AllNull)),
    ];
    let rows = [
        DeepOptionalBoolRow { values: None },
        DeepOptionalBoolRow {
            values: Some(vec![]),
        },
        DeepOptionalBoolRow {
            values: Some(expected_segments.clone()),
        },
    ];
    let dataframe = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(dataframe.shape(), (3, 1));
    assert_eq!(
        dataframe.schema().get("values"),
        Some(&DataType::List(Box::new(DataType::List(Box::new(
            DataType::Boolean,
        ))))),
    );
    assert!(
        dataframe
            .column("values")
            .unwrap()
            .get(0)
            .unwrap()
            .is_null()
    );
    let AnyValue::List(empty) = dataframe.column("values").unwrap().get(1).unwrap() else {
        panic!("expected an empty outer list");
    };
    assert!(empty.is_empty());

    let AnyValue::List(outer) = dataframe.column("values").unwrap().get(2).unwrap() else {
        panic!("expected the populated outer list");
    };
    assert_eq!(outer.len(), expected_segments.len());
    for (index, expected) in expected_segments.iter().enumerate() {
        let actual = outer.get(index).unwrap();
        let Some(expected) = expected else {
            assert!(actual.is_null(), "segment={index} actual={actual:?}");
            continue;
        };
        let AnyValue::List(inner) = actual else {
            panic!("expected inner list at segment {index}, got {actual:?}");
        };
        let actual: Vec<Option<bool>> = inner.bool().unwrap().iter().collect();
        assert_eq!(actual, *expected, "segment={index}");
    }
}

#[test]
fn fallible_numeric_validity_grows_and_trims_at_word_boundaries() {
    for len in BOUNDARY_LENGTHS {
        for pattern in ValidityPattern::ALL {
            let expected: Vec<Option<Decimal>> = (0..len)
                .map(|index| {
                    (!pattern.is_null(index))
                        .then(|| Decimal::from(i64::try_from(index).expect("test length fits i64")))
                })
                .collect();
            let rows = [OptionalDecimalRow { values: expected }];
            let dataframe = OptionalDecimalRow::encode(rows.iter()).unwrap();
            let AnyValue::List(inner) = dataframe.column("values").unwrap().get(0).unwrap() else {
                panic!("expected decimal list for len={len} pattern={pattern:?}");
            };

            assert_eq!(inner.len(), len, "len={len} pattern={pattern:?}");
            for index in 0..len {
                match inner.get(index).unwrap() {
                    AnyValue::Null if pattern.is_null(index) => {}
                    AnyValue::Decimal(mantissa, _, 6) if !pattern.is_null(index) => {
                        assert_eq!(
                            mantissa,
                            i128::try_from(index).unwrap() * 1_000_000,
                            "index={index} len={len} pattern={pattern:?}",
                        );
                    }
                    actual => panic!(
                        "unexpected value at index={index} len={len} pattern={pattern:?}: \
                         {actual:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn fallible_numeric_validity_grows_across_segments_before_final_trim() {
    let expected: Vec<Vec<Option<Decimal>>> = [1, 5, 17, 65]
        .into_iter()
        .enumerate()
        .map(|(row, len)| {
            (0..len)
                .map(|index| {
                    ((row + index) % 4 != 0)
                        .then(|| Decimal::from(i64::try_from(row * 100 + index).unwrap()))
                })
                .collect()
        })
        .collect();
    let rows: Vec<OptionalDecimalRow> = expected
        .iter()
        .cloned()
        .map(|values| OptionalDecimalRow { values })
        .collect();
    let dataframe = OptionalDecimalRow::encode(rows.iter()).unwrap();

    assert_eq!(dataframe.height(), expected.len());
    for (row, expected) in expected.iter().enumerate() {
        let AnyValue::List(inner) = dataframe.column("values").unwrap().get(row).unwrap() else {
            panic!("expected decimal list at row {row}");
        };
        assert_eq!(inner.len(), expected.len(), "row={row}");
        assert_eq!(
            inner.null_count(),
            expected.iter().filter(|value| value.is_none()).count(),
            "row={row}",
        );
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(
                inner.get(index).unwrap().is_null(),
                expected.is_none(),
                "row={row} index={index}",
            );
        }
    }
}

#[test]
fn fallible_numeric_lists_stop_before_later_source_rows() {
    let yielded = Cell::new(0);
    let rows = [
        OptionalDecimalRow {
            values: vec![Some(Decimal::MAX)],
        },
        OptionalDecimalRow {
            values: vec![Some(Decimal::MAX)],
        },
    ];
    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));

    let error = OptionalDecimalRow::encode(inspected).unwrap_err();
    assert!(
        error.to_string().contains("df-derive: decimal mantissa"),
        "unexpected error: {error}",
    );
    assert_eq!(yielded.get(), 1);
}
