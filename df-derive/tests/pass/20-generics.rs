use df_derive::ToDataFrame;
use polars::prelude::*;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};

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

// Generic field wrapped in Option
#[derive(ToDataFrame, Clone)]
struct OptWrapper<T>
where
    T: Clone,
{
    id: u32,
    payload: Option<T>,
}

// Generic field wrapped in Vec
#[derive(ToDataFrame, Clone)]
struct VecWrapper<T>
where
    T: Clone,
{
    id: u32,
    payload: Vec<T>,
}

// `Option<Option<T>>`: `Some(None)` and `None` are the same null, for a
// generic payload as for a concrete one.
#[derive(ToDataFrame, Clone)]
struct OptOptWrapper<T>
where
    T: Clone,
{
    id: u32,
    payload: Option<Option<T>>,
}

// `Option<Vec<T>>`: a nullable list of a generic payload.
#[derive(ToDataFrame, Clone)]
struct OptVecWrapper<T>
where
    T: Clone,
{
    id: u32,
    payload: Option<Vec<T>>,
}

// `Vec<Option<T>>`: a list of nullable generic payloads.
#[derive(ToDataFrame, Clone)]
struct VecOptWrapper<T>
where
    T: Clone,
{
    id: u32,
    payload: Vec<Option<T>>,
}

// `Vec<Vec<T>>`: a list of lists of a generic payload.
#[derive(ToDataFrame, Clone)]
struct VecVecWrapper<T>
where
    T: Clone,
{
    id: u32,
    payload: Vec<Vec<T>>,
}

#[derive(ToDataFrame, Clone)]
struct InnerGeneric<M>
where
    M: Clone,
{
    label: String,
    payload: M,
}

// Outer struct that propagates its own `M` into nested generic fields.
#[derive(ToDataFrame, Clone)]
struct OuterPropagating<M = ()>
where
    M: Clone,
{
    id: u32,
    direct: InnerGeneric<M>,
    optional: Option<InnerGeneric<M>>,
    listed: Vec<InnerGeneric<M>>,
}

fn main() {
    test_primitive_instantiation();
    test_nested_struct_instantiation();
    test_unit_instantiation();
    test_default_type_parameter();
    test_multiple_generics();
    test_option_wrapped_generic();
    test_vec_wrapped_generic();
    test_doubly_wrapped_generic();
    test_depth2_combos();
    test_propagated_type_parameter();
    println!("All generics tests passed!");
}

fn test_primitive_instantiation() {
    println!("Testing primitive instantiation (Wrapper<f64>)...");

    let w: Wrapper<f64> = Wrapper { id: 1, payload: 3.5 };

    // schema/columns: id (u32) + payload.value (f64)
    let schema = Wrapper::<f64>::schema().unwrap();
    assert_eq!(
        schema,
        vec![
            ("id".into(), DataType::UInt32),
            ("payload".into(), DataType::Float64),
        ]
    );

    let df = w.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 2));
    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt32(1)
    );
    assert_eq!(
        df.column("payload").unwrap().get(0).unwrap(),
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
        batch_df.column("payload").unwrap().get(2).unwrap(),
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
            ("id".into(), DataType::UInt32),
            ("payload.timestamp".into(), DataType::Int64),
            ("payload.note".into(), DataType::String),
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
    assert_eq!(schema, vec![("id".into(), DataType::UInt32)]);

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
    assert_eq!(schema, vec![("val".into(), DataType::Int32)]);

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
            ("a.timestamp".into(), DataType::Int64),
            ("a.note".into(), DataType::String),
            ("name".into(), DataType::String),
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
            ("a.timestamp".into(), DataType::Int64),
            ("a.note".into(), DataType::String),
            ("b.timestamp".into(), DataType::Int64),
            ("b.note".into(), DataType::String),
            ("name".into(), DataType::String),
        ]
    );

    let pair_df = pair.to_dataframe().unwrap();
    assert_eq!(pair_df.shape(), (1, 5));
    assert_eq!(
        pair_df.column("b.note").unwrap().get(0).unwrap(),
        AnyValue::String("rhs")
    );
}

fn test_option_wrapped_generic() {
    println!("Testing Option<T> for generic T...");

    // Option<f64>: scalar primitive payload, optional.
    let schema = OptWrapper::<f64>::schema().unwrap();
    assert_eq!(
        schema,
        vec![
            ("id".into(), DataType::UInt32),
            ("payload".into(), DataType::Float64),
        ]
    );

    let some_w = OptWrapper {
        id: 1,
        payload: Some(3.5_f64),
    };
    let none_w: OptWrapper<f64> = OptWrapper { id: 2, payload: None };

    let df_some = some_w.to_dataframe().unwrap();
    assert_eq!(df_some.shape(), (1, 2));
    assert_eq!(
        df_some.column("payload").unwrap().get(0).unwrap(),
        AnyValue::Float64(3.5)
    );

    let df_none = none_w.to_dataframe().unwrap();
    assert_eq!(df_none.shape(), (1, 2));
    assert!(matches!(
        df_none.column("payload").unwrap().get(0).unwrap(),
        AnyValue::Null
    ));

    // Batch: Some, None, Some
    let items = vec![
        OptWrapper {
            id: 10,
            payload: Some(1.0_f64),
        },
        OptWrapper {
            id: 11,
            payload: None,
        },
        OptWrapper {
            id: 12,
            payload: Some(2.0_f64),
        },
    ];
    let batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape(), (3, 2));
    assert_eq!(
        batch.column("payload").unwrap().get(0).unwrap(),
        AnyValue::Float64(1.0)
    );
    assert!(matches!(
        batch.column("payload").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert_eq!(
        batch.column("payload").unwrap().get(2).unwrap(),
        AnyValue::Float64(2.0)
    );

    // Option<MetaStruct>: nested struct payload, optional.
    let schema_meta = OptWrapper::<MetaStruct>::schema().unwrap();
    assert_eq!(
        schema_meta,
        vec![
            ("id".into(), DataType::UInt32),
            ("payload.timestamp".into(), DataType::Int64),
            ("payload.note".into(), DataType::String),
        ]
    );

    let items_meta = vec![
        OptWrapper {
            id: 1,
            payload: Some(MetaStruct {
                timestamp: 100,
                note: "a".to_string(),
            }),
        },
        OptWrapper {
            id: 2,
            payload: None,
        },
        OptWrapper {
            id: 3,
            payload: Some(MetaStruct {
                timestamp: 300,
                note: "c".to_string(),
            }),
        },
    ];
    let batch_meta = items_meta.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_meta.shape(), (3, 3));
    assert_eq!(
        batch_meta
            .column("payload.timestamp")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int64(100)
    );
    assert!(matches!(
        batch_meta
            .column("payload.timestamp")
            .unwrap()
            .get(1)
            .unwrap(),
        AnyValue::Null
    ));
    assert_eq!(
        batch_meta.column("payload.note").unwrap().get(2).unwrap(),
        AnyValue::String("c")
    );

    let empty = OptWrapper::<MetaStruct>::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 3));
}

fn test_vec_wrapped_generic() {
    println!("Testing Vec<T> for generic T...");

    // Vec<f64>: list of primitive payload.
    let schema = VecWrapper::<f64>::schema().unwrap();
    assert_eq!(
        schema,
        vec![
            ("id".into(), DataType::UInt32),
            (
                "payload".into(),
                DataType::List(Box::new(DataType::Float64)),
            ),
        ]
    );

    let items = vec![
        VecWrapper {
            id: 1,
            payload: vec![1.0_f64, 2.0],
        },
        VecWrapper {
            id: 2,
            payload: vec![],
        },
        VecWrapper {
            id: 3,
            payload: vec![3.0],
        },
    ];
    let df = items.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (3, 2));
    assert_eq!(
        df.column("payload").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Float64))
    );

    // Each row's "payload" is itself a list. Materialize and check.
    let row0 = df.column("payload").unwrap().get(0).unwrap();
    let row1 = df.column("payload").unwrap().get(1).unwrap();
    let row2 = df.column("payload").unwrap().get(2).unwrap();
    assert!(matches!(row0, AnyValue::List(_)));
    assert!(matches!(row1, AnyValue::List(_)));
    assert!(matches!(row2, AnyValue::List(_)));
    if let AnyValue::List(s) = row0 {
        assert_eq!(s.len(), 2);
    }
    if let AnyValue::List(s) = row1 {
        assert_eq!(s.len(), 0);
    }
    if let AnyValue::List(s) = row2 {
        assert_eq!(s.len(), 1);
    }

    // Vec<MetaStruct>: list of nested struct.
    let schema_meta = VecWrapper::<MetaStruct>::schema().unwrap();
    assert_eq!(
        schema_meta,
        vec![
            ("id".into(), DataType::UInt32),
            (
                "payload.timestamp".into(),
                DataType::List(Box::new(DataType::Int64)),
            ),
            (
                "payload.note".into(),
                DataType::List(Box::new(DataType::String)),
            ),
        ]
    );

    let items_meta = vec![
        VecWrapper {
            id: 1,
            payload: vec![
                MetaStruct {
                    timestamp: 1,
                    note: "a".to_string(),
                },
                MetaStruct {
                    timestamp: 2,
                    note: "b".to_string(),
                },
            ],
        },
        VecWrapper {
            id: 2,
            payload: vec![],
        },
    ];
    let df_meta = items_meta.as_slice().to_dataframe().unwrap();
    assert_eq!(df_meta.shape(), (2, 3));
    let ts_row0 = df_meta.column("payload.timestamp").unwrap().get(0).unwrap();
    assert!(matches!(ts_row0, AnyValue::List(_)));
    if let AnyValue::List(s) = ts_row0 {
        assert_eq!(s.len(), 2);
    }
    let ts_row1 = df_meta.column("payload.timestamp").unwrap().get(1).unwrap();
    if let AnyValue::List(s) = ts_row1 {
        assert_eq!(s.len(), 0);
    }

    // Single-instance to_dataframe.
    let single = VecWrapper {
        id: 99,
        payload: vec![42.0_f64, 43.0, 44.0],
    };
    let single_df = single.to_dataframe().unwrap();
    assert_eq!(single_df.shape(), (1, 2));
    if let AnyValue::List(s) = single_df.column("payload").unwrap().get(0).unwrap() {
        assert_eq!(s.len(), 3);
    } else {
        panic!("expected List AnyValue");
    }

    let empty = VecWrapper::<f64>::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 2));
}

fn test_doubly_wrapped_generic() {
    println!("Testing Option<Option<T>> for generic T...");

    // `Some(None)` and `None` cannot be told apart in the frame: both are one
    // null. The schema is that of `T`. That is the documented contract.
    let schema = OptOptWrapper::<f64>::schema().unwrap();
    assert_eq!(
        schema,
        vec![
            ("id".into(), DataType::UInt32),
            ("payload".into(), DataType::Float64),
        ]
    );

    let items = vec![
        OptOptWrapper {
            id: 1,
            payload: Some(Some(7.5_f64)),
        },
        OptOptWrapper {
            id: 2,
            payload: Some(None),
        },
        OptOptWrapper {
            id: 3,
            payload: None,
        },
    ];
    let batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape(), (3, 2));
    assert_eq!(
        batch.column("payload").unwrap().get(0).unwrap(),
        AnyValue::Float64(7.5)
    );
    assert!(matches!(
        batch.column("payload").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert!(matches!(
        batch.column("payload").unwrap().get(2).unwrap(),
        AnyValue::Null
    ));
}

fn test_depth2_combos() {
    println!("Testing Option<Vec<T>> / Vec<Option<T>> / Vec<Vec<T>> for generic T...");

    // Option<Vec<T>> with T = f64.
    let schema_ov = OptVecWrapper::<f64>::schema().unwrap();
    assert_eq!(
        schema_ov,
        vec![
            ("id".into(), DataType::UInt32),
            (
                "payload".into(),
                DataType::List(Box::new(DataType::Float64)),
            ),
        ]
    );
    let items_ov = vec![
        OptVecWrapper {
            id: 1,
            payload: Some(vec![1.0_f64, 2.0]),
        },
        OptVecWrapper { id: 2, payload: None },
        OptVecWrapper {
            id: 3,
            payload: Some(vec![]),
        },
    ];
    let df_ov = items_ov.as_slice().to_dataframe().unwrap();
    assert_eq!(df_ov.shape(), (3, 2));
    let row0 = df_ov.column("payload").unwrap().get(0).unwrap();
    let row1 = df_ov.column("payload").unwrap().get(1).unwrap();
    let row2 = df_ov.column("payload").unwrap().get(2).unwrap();
    if let AnyValue::List(s) = row0 {
        assert_eq!(s.len(), 2);
    } else {
        panic!("expected List for Some(non-empty)");
    }
    assert!(matches!(row1, AnyValue::Null | AnyValue::List(_)));
    if let AnyValue::List(s) = row2 {
        assert_eq!(s.len(), 0);
    } else {
        panic!("expected List for Some(empty)");
    }

    // Vec<Option<T>> with T = f64.
    let schema_vo = VecOptWrapper::<f64>::schema().unwrap();
    assert_eq!(
        schema_vo,
        vec![
            ("id".into(), DataType::UInt32),
            (
                "payload".into(),
                DataType::List(Box::new(DataType::Float64)),
            ),
        ]
    );
    let items_vo = vec![
        VecOptWrapper {
            id: 1,
            payload: vec![Some(1.0_f64), None, Some(3.0)],
        },
        VecOptWrapper {
            id: 2,
            payload: vec![],
        },
    ];
    let df_vo = items_vo.as_slice().to_dataframe().unwrap();
    assert_eq!(df_vo.shape(), (2, 2));
    let row0 = df_vo.column("payload").unwrap().get(0).unwrap();
    if let AnyValue::List(s) = row0 {
        assert_eq!(s.len(), 3);
        // Middle element of the inner list should be null.
        assert!(matches!(s.get(1).unwrap(), AnyValue::Null));
    } else {
        panic!("expected List for Vec<Option<T>>");
    }

    // Vec<Vec<T>> with T = f64. The schema generator wraps once per `Vec`
    // layer, so the declared dtype matches the runtime List<List<...>>.
    let schema_vv = VecVecWrapper::<f64>::schema().unwrap();
    assert_eq!(schema_vv.len(), 2);
    assert_eq!(schema_vv[0], ("id".into(), DataType::UInt32));
    assert_eq!(
        schema_vv[1],
        (
            "payload".into(),
            DataType::List(Box::new(DataType::List(Box::new(DataType::Float64))))
        )
    );
    let items_vv = vec![
        VecVecWrapper {
            id: 1,
            payload: vec![vec![1.0_f64, 2.0], vec![3.0]],
        },
        VecVecWrapper {
            id: 2,
            payload: vec![],
        },
    ];
    let df_vv = items_vv.as_slice().to_dataframe().unwrap();
    assert_eq!(df_vv.shape(), (2, 2));
    let row0 = df_vv.column("payload").unwrap().get(0).unwrap();
    if let AnyValue::List(s) = row0 {
        // Outer list has 2 inner lists.
        assert_eq!(s.len(), 2);
    } else {
        panic!("expected outer List for Vec<Vec<T>>");
    }
}

fn test_propagated_type_parameter() {
    println!("Testing outer struct propagating its type param into nested generics...");

    let item_f64 = OuterPropagating::<f64> {
        id: 7,
        direct: InnerGeneric {
            label: "d".into(),
            payload: 1.5,
        },
        optional: Some(InnerGeneric {
            label: "o".into(),
            payload: 2.5,
        }),
        listed: vec![
            InnerGeneric {
                label: "a".into(),
                payload: 3.5,
            },
            InnerGeneric {
                label: "b".into(),
                payload: 4.5,
            },
        ],
    };

    let schema = OuterPropagating::<f64>::schema().unwrap();
    let names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"id"));
    assert!(names.contains(&"direct.label"));
    assert!(names.contains(&"direct.payload"));
    assert!(names.contains(&"optional.label"));
    assert!(names.contains(&"optional.payload"));
    assert!(names.contains(&"listed.label"));
    assert!(names.contains(&"listed.payload"));

    let df = item_f64.to_dataframe().unwrap();
    assert_eq!(df.shape().0, 1);

    let items = vec![
        item_f64,
        OuterPropagating::<f64> {
            id: 8,
            direct: InnerGeneric {
                label: "d2".into(),
                payload: 11.0,
            },
            optional: None,
            listed: vec![],
        },
    ];
    let df_batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape().0, 2);

    let item_unit: OuterPropagating = OuterPropagating {
        id: 1,
        direct: InnerGeneric {
            label: "u".into(),
            payload: (),
        },
        optional: None,
        listed: vec![],
    };
    let _df_unit = item_unit.to_dataframe().unwrap();
}
