use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{Columnar, ToDataFrame, ToDataFrameVec};

// Nested struct used as a generic instantiation target
#[derive(ToDataFrame, Clone)]
struct MetaStruct {
    timestamp: i64,
    note: String,
}

// Generic struct with one type parameter
#[derive(ToDataFrame, Clone)]
struct Wrapper<T>
where
    T: Clone,
{
    id: u32,
    payload: T,
}

// Default type parameter (M defaults to ())
#[derive(ToDataFrame, Clone)]
struct DefaultMeta<M = ()>
where
    M: Clone,
{
    val: i32,
    meta: M,
}

// Multiple generics
#[derive(ToDataFrame, Clone)]
struct Multi<A, B>
where
    A: Clone,
    B: Clone,
{
    a: A,
    b: B,
    name: String,
}

// Local impls of ToDataFrame/Columnar for f64 so generic instantiation with a
// primitive can flatten via a single column. Implementing on a foreign primitive
// is allowed because the trait is defined in this test crate.
impl ToDataFrame for f64 {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        DataFrame::new(vec![Series::new("value".into(), &[*self]).into()])
    }
    fn empty_dataframe() -> PolarsResult<DataFrame> {
        DataFrame::new(vec![Series::new_empty("value".into(), &DataType::Float64).into()])
    }
    fn schema() -> PolarsResult<Vec<(&'static str, DataType)>> {
        Ok(vec![("value", DataType::Float64)])
    }
}

impl Columnar for f64 {
    fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
        DataFrame::new(vec![Series::new("value".into(), items).into()])
    }
}

fn main() {
    test_primitive_instantiation();
    test_nested_struct_instantiation();
    test_unit_instantiation();
    test_default_type_parameter();
    test_multiple_generics();
    println!("All generics tests passed!");
}

fn test_primitive_instantiation() {
    println!("Testing primitive instantiation (Wrapper<f64>)...");

    let w: Wrapper<f64> = Wrapper { id: 1, payload: 3.5 };

    // schema/columns: id (u32) + payload.value (f64)
    let schema = Wrapper::<f64>::schema().unwrap();
    assert_eq!(
        schema,
        vec![("id", DataType::UInt32), ("payload.value", DataType::Float64)]
    );

    let df = w.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 2));
    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt32(1)
    );
    assert_eq!(
        df.column("payload.value").unwrap().get(0).unwrap(),
        AnyValue::Float64(3.5)
    );

    let empty = Wrapper::<f64>::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 2));

    let items = vec![
        Wrapper { id: 1, payload: 1.0 },
        Wrapper { id: 2, payload: 2.0 },
        Wrapper { id: 3, payload: 3.0 },
    ];
    let batch_df = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.shape(), (3, 2));
    assert_eq!(
        batch_df.column("id").unwrap().get(2).unwrap(),
        AnyValue::UInt32(3)
    );
    assert_eq!(
        batch_df.column("payload.value").unwrap().get(2).unwrap(),
        AnyValue::Float64(3.0)
    );

    // Empty slice should round-trip through ToDataFrameVec to empty_dataframe.
    let empty_slice: &[Wrapper<f64>] = &[];
    let empty_batch = empty_slice.to_dataframe().unwrap();
    assert_eq!(empty_batch.shape(), (0, 2));
}

fn test_nested_struct_instantiation() {
    println!("Testing nested struct instantiation (Wrapper<MetaStruct>)...");

    let w = Wrapper {
        id: 42,
        payload: MetaStruct {
            timestamp: 1_700_000_000,
            note: "hello".to_string(),
        },
    };

    let schema = Wrapper::<MetaStruct>::schema().unwrap();
    assert_eq!(
        schema,
        vec![
            ("id", DataType::UInt32),
            ("payload.timestamp", DataType::Int64),
            ("payload.note", DataType::String),
        ]
    );

    let df = w.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 3));
    assert_eq!(
        df.column("payload.timestamp").unwrap().get(0).unwrap(),
        AnyValue::Int64(1_700_000_000)
    );
    assert_eq!(
        df.column("payload.note").unwrap().get(0).unwrap(),
        AnyValue::String("hello")
    );

    let empty = Wrapper::<MetaStruct>::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 3));

    let items = vec![
        Wrapper {
            id: 1,
            payload: MetaStruct {
                timestamp: 10,
                note: "a".to_string(),
            },
        },
        Wrapper {
            id: 2,
            payload: MetaStruct {
                timestamp: 20,
                note: "b".to_string(),
            },
        },
    ];
    let batch_df = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.shape(), (2, 3));
    assert_eq!(
        batch_df.column("payload.timestamp").unwrap().get(1).unwrap(),
        AnyValue::Int64(20)
    );
    assert_eq!(
        batch_df.column("payload.note").unwrap().get(1).unwrap(),
        AnyValue::String("b")
    );
}

fn test_unit_instantiation() {
    println!("Testing unit instantiation (Wrapper<()>)...");

    let w: Wrapper<()> = Wrapper { id: 7, payload: () };

    let schema = Wrapper::<()>::schema().unwrap();
    assert_eq!(schema, vec![("id", DataType::UInt32)]);

    let df = w.to_dataframe().unwrap();
    // Unit payload contributes zero columns; id remains.
    assert_eq!(df.shape(), (1, 1));
    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt32(7)
    );

    let empty = Wrapper::<()>::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 1));

    let items: Vec<Wrapper<()>> = (0..5)
        .map(|i| Wrapper { id: i, payload: () })
        .collect();
    let batch_df = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.shape(), (5, 1));
    assert_eq!(
        batch_df.column("id").unwrap().get(4).unwrap(),
        AnyValue::UInt32(4)
    );
}

fn test_default_type_parameter() {
    println!("Testing default type parameter (DefaultMeta with no arg)...");

    // Without specifying M, it defaults to ().
    let dm = DefaultMeta { val: 99, meta: () };

    let schema = DefaultMeta::<()>::schema().unwrap();
    assert_eq!(schema, vec![("val", DataType::Int32)]);

    let df = dm.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 1));
    assert_eq!(
        df.column("val").unwrap().get(0).unwrap(),
        AnyValue::Int32(99)
    );

    let empty = DefaultMeta::<()>::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 1));

    // Now override the default with a concrete struct.
    let dm2 = DefaultMeta {
        val: 10,
        meta: MetaStruct {
            timestamp: 5,
            note: "x".to_string(),
        },
    };
    let df2 = dm2.to_dataframe().unwrap();
    assert_eq!(df2.shape(), (1, 3));
    assert_eq!(
        df2.column("meta.timestamp").unwrap().get(0).unwrap(),
        AnyValue::Int64(5)
    );

    let items = vec![
        DefaultMeta { val: 1, meta: () },
        DefaultMeta { val: 2, meta: () },
        DefaultMeta { val: 3, meta: () },
    ];
    let batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape(), (3, 1));
    assert_eq!(
        batch.column("val").unwrap().get(0).unwrap(),
        AnyValue::Int32(1)
    );
}

fn test_multiple_generics() {
    println!("Testing multiple generic parameters (Multi<A, B>)...");

    // A = MetaStruct, B = ()
    let m = Multi {
        a: MetaStruct {
            timestamp: 100,
            note: "n".to_string(),
        },
        b: (),
        name: "row".to_string(),
    };

    let schema = Multi::<MetaStruct, ()>::schema().unwrap();
    assert_eq!(
        schema,
        vec![
            ("a.timestamp", DataType::Int64),
            ("a.note", DataType::String),
            ("name", DataType::String),
        ]
    );

    let df = m.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 3));
    assert_eq!(
        df.column("a.timestamp").unwrap().get(0).unwrap(),
        AnyValue::Int64(100)
    );
    assert_eq!(
        df.column("a.note").unwrap().get(0).unwrap(),
        AnyValue::String("n")
    );
    assert_eq!(
        df.column("name").unwrap().get(0).unwrap(),
        AnyValue::String("row")
    );

    let empty = Multi::<MetaStruct, ()>::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 3));

    let items = vec![
        Multi {
            a: MetaStruct {
                timestamp: 1,
                note: "x".to_string(),
            },
            b: (),
            name: "r1".to_string(),
        },
        Multi {
            a: MetaStruct {
                timestamp: 2,
                note: "y".to_string(),
            },
            b: (),
            name: "r2".to_string(),
        },
    ];
    let batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape(), (2, 3));
    assert_eq!(
        batch.column("a.timestamp").unwrap().get(1).unwrap(),
        AnyValue::Int64(2)
    );
    assert_eq!(
        batch.column("name").unwrap().get(1).unwrap(),
        AnyValue::String("r2")
    );

    // Also try a Multi where both generics are real custom structs.
    let pair = Multi {
        a: MetaStruct {
            timestamp: 1,
            note: "lhs".to_string(),
        },
        b: MetaStruct {
            timestamp: 2,
            note: "rhs".to_string(),
        },
        name: "pair".to_string(),
    };

    let schema_pair = Multi::<MetaStruct, MetaStruct>::schema().unwrap();
    assert_eq!(
        schema_pair,
        vec![
            ("a.timestamp", DataType::Int64),
            ("a.note", DataType::String),
            ("b.timestamp", DataType::Int64),
            ("b.note", DataType::String),
            ("name", DataType::String),
        ]
    );

    let pair_df = pair.to_dataframe().unwrap();
    assert_eq!(pair_df.shape(), (1, 5));
    assert_eq!(
        pair_df.column("b.note").unwrap().get(0).unwrap(),
        AnyValue::String("rhs")
    );
}
