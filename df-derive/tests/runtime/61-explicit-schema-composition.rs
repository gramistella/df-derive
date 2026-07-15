use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use df_derive::ToDataFrame;
use df_derive::dataframe::{ColumnSink, ColumnarSpec, RowCursor, ToDataFrame as _};
use polars::prelude::*;

static CHILD_ENCODINGS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct CountingInner {
    value: u32,
}

impl ColumnarSpec for CountingInner {
    fn build_schema() -> PolarsResult<SchemaRef> {
        Ok(Arc::new(Schema::from_iter_check_duplicates([(
            "value".into(),
            DataType::UInt32,
        )])?))
    }

    fn encode_columns<'a, I>(rows: &mut I, sink: &mut ColumnSink) -> PolarsResult<()>
    where
        Self: 'a,
        I: RowCursor<Item = &'a Self>,
    {
        CHILD_ENCODINGS.fetch_add(1, Ordering::SeqCst);
        let values: Vec<u32> = rows.map(|row| row.value).collect();
        let slot = sink.next_slot()?;
        let column = Series::new(slot.name().clone(), values);
        slot.commit(column.into())
    }
}

#[derive(ToDataFrame)]
struct DirectOuter {
    inner: CountingInner,
}

#[derive(ToDataFrame)]
struct OptionalOuter {
    inner: Option<CountingInner>,
}

#[derive(ToDataFrame)]
struct VecOuter {
    inner: Vec<CountingInner>,
}

#[test]
fn runtime_semantics() -> PolarsResult<()> {
    CHILD_ENCODINGS.store(0, Ordering::SeqCst);

    let schema = DirectOuter::schema()?;
    assert_eq!(schema.get("inner.value"), Some(&DataType::UInt32));
    assert_eq!(CHILD_ENCODINGS.load(Ordering::SeqCst), 0);

    let empty = DirectOuter::empty_dataframe()?;
    assert_eq!(empty.shape(), (0, 1));
    assert_eq!(CHILD_ENCODINGS.load(Ordering::SeqCst), 0);

    let direct = [DirectOuter {
        inner: CountingInner { value: 7 },
    }];
    let direct = df_derive::dataframe::Columnar::encode(direct.as_slice())?;
    assert_eq!(direct.shape(), (1, 1));
    assert_eq!(CHILD_ENCODINGS.load(Ordering::SeqCst), 1);

    let absent = [OptionalOuter { inner: None }, OptionalOuter { inner: None }];
    let absent = df_derive::dataframe::Columnar::encode(absent.as_slice())?;
    assert_eq!(absent.shape(), (2, 1));
    assert_eq!(CHILD_ENCODINGS.load(Ordering::SeqCst), 1);

    let empty_lists = [
        VecOuter { inner: Vec::new() },
        VecOuter { inner: Vec::new() },
    ];
    let empty_lists = df_derive::dataframe::Columnar::encode(empty_lists.as_slice())?;
    assert_eq!(empty_lists.shape(), (2, 1));
    assert_eq!(CHILD_ENCODINGS.load(Ordering::SeqCst), 1);

    let populated = [VecOuter {
        inner: vec![CountingInner { value: 11 }, CountingInner { value: 13 }],
    }];
    let populated = df_derive::dataframe::Columnar::encode(populated.as_slice())?;
    assert_eq!(populated.shape(), (1, 1));
    assert_eq!(CHILD_ENCODINGS.load(Ordering::SeqCst), 2);

    Ok(())
}
