use crate::core::dataframe::{Columnar, ToDataFrame};
use df_derive::ToDataFrame;
use polars::prelude::*;

#[derive(Clone)]
struct BadHeightInner {
    _value: i64,
}

#[derive(Clone)]
struct BadDtypeInner {
    _value: i64,
}

impl ToDataFrame for BadHeightInner {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        <Self as Columnar>::columnar_from_refs(&[self])
    }

    fn empty_dataframe() -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![
            Series::new_empty("value".into(), &DataType::Int64).into(),
        ])
    }

    fn schema() -> PolarsResult<Vec<(String, DataType)>> {
        Ok(vec![("value".into(), DataType::Int64)])
    }
}

impl Columnar for BadHeightInner {
    fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
        let values: Vec<i64> = (0..=items.len() as i64).collect();
        DataFrame::new_infer_height(vec![Series::new("value".into(), values).into()])
    }
}

impl ToDataFrame for BadDtypeInner {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        <Self as Columnar>::columnar_from_refs(&[self])
    }

    fn empty_dataframe() -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![
            Series::new_empty("value".into(), &DataType::Int64).into(),
        ])
    }

    fn schema() -> PolarsResult<Vec<(String, DataType)>> {
        Ok(vec![("value".into(), DataType::Int64)])
    }
}

impl Columnar for BadDtypeInner {
    fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
        let values: Vec<String> = items
            .iter()
            .enumerate()
            .map(|(i, _)| format!("wrong-{i}"))
            .collect();
        DataFrame::new_infer_height(vec![Series::new("value".into(), values).into()])
    }
}

#[derive(ToDataFrame, Clone)]
struct DirectHeightOuter {
    inner: BadHeightInner,
}

#[derive(ToDataFrame, Clone)]
struct DirectDtypeOuter {
    inner: BadDtypeInner,
}

#[derive(ToDataFrame, Clone)]
struct TupleHeightOuter {
    payload: Vec<(BadHeightInner,)>,
}

#[derive(ToDataFrame, Clone)]
struct TupleDtypeOuter {
    payload: Vec<(BadDtypeInner,)>,
}

fn assert_compute_error_contains(result: PolarsResult<DataFrame>, expected: &str) {
    let Err(err) = result else {
        panic!("expected ComputeError containing `{expected}`");
    };
    match err {
        PolarsError::ComputeError(msg) => assert!(
            msg.contains(expected),
            "unexpected ComputeError message: {msg}"
        ),
        other => panic!("expected ComputeError containing `{expected}`, got {other:?}"),
    }
}

#[test]
fn runtime_semantics() {
    assert_compute_error_contains(
        <DirectHeightOuter as Columnar>::columnar_to_dataframe(&[DirectHeightOuter {
            inner: BadHeightInner { _value: 1 },
        }]),
        "returned height",
    );

    assert_compute_error_contains(
        <DirectDtypeOuter as Columnar>::columnar_to_dataframe(&[DirectDtypeOuter {
            inner: BadDtypeInner { _value: 1 },
        }]),
        "dtype mismatch",
    );

    assert_compute_error_contains(
        <TupleHeightOuter as Columnar>::columnar_to_dataframe(&[TupleHeightOuter {
            payload: vec![(BadHeightInner { _value: 1 },)],
        }]),
        "returned height",
    );

    assert_compute_error_contains(
        <TupleDtypeOuter as Columnar>::columnar_to_dataframe(&[TupleDtypeOuter {
            payload: vec![(BadDtypeInner { _value: 1 },)],
        }]),
        "dtype mismatch",
    );
}
