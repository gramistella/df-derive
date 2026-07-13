use df_derive::ToDataFrame;
use df_derive::dataframe::Columnar;
use polars::prelude::*;

#[derive(Clone)]
struct BadHeightInner;

#[derive(Clone)]
struct ExtraColumnInner;

#[derive(Clone)]
struct MissingColumnInner;

#[derive(Clone)]
struct ReorderedColumnsInner;

#[derive(Clone)]
struct BadDtypeInner;

#[derive(Clone)]
struct ValidInner {
    value: i64,
    label: String,
}

fn empty_value_frame() -> PolarsResult<DataFrame> {
    DataFrame::new(
        0,
        vec![Series::new_empty("value".into(), &DataType::Int64).into()],
    )
}

fn empty_value_label_frame() -> PolarsResult<DataFrame> {
    DataFrame::new(
        0,
        vec![
            Series::new_empty("value".into(), &DataType::Int64).into(),
            Series::new_empty("label".into(), &DataType::String).into(),
        ],
    )
}

impl Columnar for BadHeightInner {
    fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        R: IntoIterator<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.into_iter().collect();
        if rows.is_empty() {
            return empty_value_frame();
        }
        let values: Vec<i64> = (0..=rows.len() as i64).collect();
        DataFrame::new(
            values.len(),
            vec![Series::new("value".into(), values).into()],
        )
    }
}

impl Columnar for ExtraColumnInner {
    fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        R: IntoIterator<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.into_iter().collect();
        if rows.is_empty() {
            return empty_value_frame();
        }
        let values = vec![1_i64; rows.len()];
        let extras = vec![2_i64; rows.len()];
        DataFrame::new(
            rows.len(),
            vec![
                Series::new("value".into(), values).into(),
                Series::new("extra".into(), extras).into(),
            ],
        )
    }
}

impl Columnar for MissingColumnInner {
    fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        R: IntoIterator<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.into_iter().collect();
        if rows.is_empty() {
            return empty_value_label_frame();
        }
        let values = vec![1_i64; rows.len()];
        DataFrame::new(rows.len(), vec![Series::new("value".into(), values).into()])
    }
}

impl Columnar for ReorderedColumnsInner {
    fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        R: IntoIterator<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.into_iter().collect();
        if rows.is_empty() {
            return empty_value_label_frame();
        }
        let labels = vec![String::from("wrong-order"); rows.len()];
        let values = vec![1_i64; rows.len()];
        DataFrame::new(
            rows.len(),
            vec![
                Series::new("label".into(), labels).into(),
                Series::new("value".into(), values).into(),
            ],
        )
    }
}

impl Columnar for BadDtypeInner {
    fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        R: IntoIterator<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.into_iter().collect();
        if rows.is_empty() {
            return empty_value_frame();
        }
        let values = vec![String::from("wrong-dtype"); rows.len()];
        DataFrame::new(rows.len(), vec![Series::new("value".into(), values).into()])
    }
}

impl Columnar for ValidInner {
    fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
    where
        Self: 'a,
        R: IntoIterator<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.into_iter().collect();
        let values: Vec<i64> = rows.iter().map(|row| row.value).collect();
        let labels: Vec<&str> = rows.iter().map(|row| row.label.as_str()).collect();
        DataFrame::new(
            rows.len(),
            vec![
                Series::new("value".into(), values).into(),
                Series::new("label".into(), labels).into(),
            ],
        )
    }
}

#[derive(ToDataFrame, Clone)]
struct DirectOuter<T> {
    inner: T,
}

#[derive(ToDataFrame, Clone)]
struct TupleOuter<T> {
    payload: Vec<(T,)>,
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

fn encode_one<T>(inner: T) -> PolarsResult<DataFrame>
where
    T: Columnar,
{
    let row = DirectOuter { inner };
    <DirectOuter<T> as Columnar>::encode(std::slice::from_ref(&row))
}

fn encode_tuple<T>(inner: T) -> PolarsResult<DataFrame>
where
    T: Columnar,
{
    let row = TupleOuter {
        payload: vec![(inner,)],
    };
    <TupleOuter<T> as Columnar>::encode(std::slice::from_ref(&row))
}

#[test]
fn runtime_semantics() {
    for result in [encode_one(BadHeightInner), encode_tuple(BadHeightInner)] {
        assert_compute_error_contains(result, "returned height");
    }

    for result in [
        encode_one(ExtraColumnInner),
        encode_tuple(ExtraColumnInner),
        encode_one(MissingColumnInner),
        encode_tuple(MissingColumnInner),
    ] {
        assert_compute_error_contains(result, "schema width");
    }

    for result in [
        encode_one(ReorderedColumnsInner),
        encode_tuple(ReorderedColumnsInner),
    ] {
        assert_compute_error_contains(result, "returned column");
    }

    for result in [encode_one(BadDtypeInner), encode_tuple(BadDtypeInner)] {
        assert_compute_error_contains(result, "returned dtype");
    }

    let direct = encode_one(ValidInner {
        value: 7,
        label: "direct".into(),
    })
    .unwrap();
    assert_eq!(direct.get_column_names(), &["inner.value", "inner.label"]);
    assert_eq!(
        direct.column("inner.value").unwrap().i64().unwrap().get(0),
        Some(7)
    );

    let tuple = encode_tuple(ValidInner {
        value: 9,
        label: "tuple".into(),
    })
    .unwrap();
    assert_eq!(
        tuple.get_column_names(),
        &["payload.field_0.value", "payload.field_0.label"]
    );
    assert_eq!(
        tuple
            .column("payload.field_0.value")
            .unwrap()
            .list()
            .unwrap()
            .get_as_series(0)
            .unwrap()
            .i64()
            .unwrap()
            .get(0),
        Some(9)
    );
}
