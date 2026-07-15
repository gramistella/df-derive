use df_derive::ToDataFrame;
use df_derive::dataframe::{ColumnSink, Columnar, ColumnarSpec, RowCursor};
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
struct PartialConsumptionInner;

#[derive(Clone)]
struct ValidInner {
    value: i64,
    label: String,
}

fn value_schema() -> PolarsResult<SchemaRef> {
    Ok(std::sync::Arc::new(Schema::from_iter_check_duplicates([
        ("value".into(), DataType::Int64),
    ])?))
}

fn value_label_schema() -> PolarsResult<SchemaRef> {
    Ok(std::sync::Arc::new(Schema::from_iter_check_duplicates([
        ("value".into(), DataType::Int64),
        ("label".into(), DataType::String),
    ])?))
}

impl ColumnarSpec for BadHeightInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        value_schema()
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.collect();
        let values: Vec<i64> = (0..=rows.len() as i64).collect();
        sink.push(Series::new("value".into(), values).into())
    }
}

impl ColumnarSpec for ExtraColumnInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        value_schema()
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.collect();
        let values = vec![1_i64; rows.len()];
        let extras = vec![2_i64; rows.len()];
        sink.push(Series::new("value".into(), values).into())?;
        sink.push(Series::new("extra".into(), extras).into())
    }
}

impl ColumnarSpec for MissingColumnInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        value_label_schema()
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.collect();
        let values = vec![1_i64; rows.len()];
        sink.push(Series::new("value".into(), values).into())
    }
}

impl ColumnarSpec for ReorderedColumnsInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        value_label_schema()
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.collect();
        let labels = vec![String::from("wrong-order"); rows.len()];
        let values = vec![1_i64; rows.len()];
        sink.push(Series::new("label".into(), labels).into())?;
        sink.push(Series::new("value".into(), values).into())
    }
}

impl ColumnarSpec for BadDtypeInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        value_schema()
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.collect();
        let values = vec![String::from("wrong-dtype"); rows.len()];
        sink.push(Series::new("value".into(), values).into())
    }
}

impl ColumnarSpec for PartialConsumptionInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        value_schema()
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let values: Vec<i64> = rows.next().map(|_| 1).into_iter().collect();
        sink.push(Series::new("value".into(), values).into())
    }
}

impl ColumnarSpec for ValidInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        value_label_schema()
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        let rows: Vec<&Self> = rows.collect();
        let values: Vec<i64> = rows.iter().map(|row| row.value).collect();
        let labels: Vec<&str> = rows.iter().map(|row| row.label.as_str()).collect();
        sink.push(Series::new("value".into(), values).into())?;
        sink.push(Series::new("label".into(), labels).into())
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

fn encode_tuple_many<T>(inners: Vec<T>) -> PolarsResult<DataFrame>
where
    T: Columnar,
{
    let row = TupleOuter {
        payload: inners.into_iter().map(|inner| (inner,)).collect(),
    };
    <TupleOuter<T> as Columnar>::encode(std::slice::from_ref(&row))
}

#[test]
fn runtime_semantics() {
    for result in [
        BadHeightInner::encode([&BadHeightInner]),
        encode_one(BadHeightInner),
        encode_tuple(BadHeightInner),
    ] {
        assert_compute_error_contains(result, "returned height");
    }

    for result in [
        ExtraColumnInner::encode([&ExtraColumnInner]),
        encode_one(ExtraColumnInner),
        encode_tuple(ExtraColumnInner),
        MissingColumnInner::encode([&MissingColumnInner]),
        encode_one(MissingColumnInner),
        encode_tuple(MissingColumnInner),
    ] {
        assert_compute_error_contains(result, "schema width");
    }

    for result in [
        ReorderedColumnsInner::encode([&ReorderedColumnsInner]),
        encode_one(ReorderedColumnsInner),
        encode_tuple(ReorderedColumnsInner),
    ] {
        assert_compute_error_contains(result, "returned column");
    }

    for result in [
        BadDtypeInner::encode([&BadDtypeInner]),
        encode_one(BadDtypeInner),
        encode_tuple(BadDtypeInner),
    ] {
        assert_compute_error_contains(result, "returned dtype");
    }

    for result in [
        PartialConsumptionInner::encode([&PartialConsumptionInner, &PartialConsumptionInner]),
        encode_tuple_many(vec![PartialConsumptionInner, PartialConsumptionInner]),
    ] {
        assert_compute_error_contains(result, "returned height");
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
