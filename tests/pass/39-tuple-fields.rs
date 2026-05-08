use chrono::{NaiveDate, NaiveTime};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::sync::Arc;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

// 1. Bare tuple — every primitive shape
#[derive(ToDataFrame, Clone)]
struct BareTuple {
    pair: (String, i32),
    triple: (f64, f64, bool),
    single: (i32,),
}

// 2. Option<tuple>
#[derive(ToDataFrame, Clone)]
struct OptTuple {
    pair: Option<(String, i32)>,
}

// 3. Vec<tuple>
#[derive(ToDataFrame, Clone)]
struct VecTuple {
    pairs: Vec<(String, i32)>,
}

// 4. Vec<Option<tuple>>
#[derive(ToDataFrame, Clone)]
struct VecOptTuple {
    items: Vec<Option<(String, i32)>>,
}

// 5. Option<Vec<tuple>>
#[derive(ToDataFrame, Clone)]
struct OptVecTuple {
    pairs: Option<Vec<(String, i32)>>,
}

// 6. Tuple in tuple struct
#[derive(ToDataFrame, Clone)]
struct WithTuple(i32, (String, bool));

// 7. Smart pointer in element
#[derive(ToDataFrame, Clone)]
struct SmartPtrTuple {
    pair: (Box<i32>, String),
}

// 8. Vec<(Arc<u64>, String)> — smart pointer + Vec
#[derive(ToDataFrame, Clone)]
struct VecSmartPtrTuple {
    items: Vec<(Arc<u64>, String)>,
}

// 9. Temporal composition
#[derive(ToDataFrame, Clone)]
struct TemporalTuple {
    times: (NaiveDate, NaiveTime),
    durs: Vec<(chrono::Duration, String)>,
}

// 10. Nested tuples (no parent wrappers)
#[derive(ToDataFrame, Clone)]
struct NestedTuple {
    nested: ((i32, String), bool),
}

// 11. Regression: HashMap rejection's hint now actionable. The hint says
// "Convert to Vec<(K, V)>"; verify the converted form compiles.
#[derive(ToDataFrame, Clone)]
struct ConvertedHashMap {
    metadata: Vec<(String, String)>,
}

// 12. Tuple element of nested-struct type (no parent wrappers).
#[derive(ToDataFrame, Clone)]
struct Inner {
    a: i32,
    b: f64,
}

#[derive(ToDataFrame, Clone)]
struct WithInnerTuple {
    pair: (Inner, i32),
}

// 13. Tuple element of nested-struct type under a Vec parent.
#[derive(ToDataFrame, Clone)]
struct VecInnerTuple {
    items: Vec<(Inner, i32)>,
}

// 14. Box<tuple> — smart pointer wrapping the tuple itself.
#[derive(ToDataFrame, Clone)]
struct BoxTuple {
    pair: Box<(String, i32)>,
}

fn main() {
    // 1. Bare
    let v = BareTuple {
        pair: ("hello".to_string(), 42),
        triple: (1.5, 2.5, true),
        single: (7,),
    };
    let df = v.to_dataframe().unwrap();
    let cols = df.get_column_names();
    let expected = [
        "pair.field_0",
        "pair.field_1",
        "triple.field_0",
        "triple.field_1",
        "triple.field_2",
        "single.field_0",
    ];
    assert_eq!(cols, expected);
    assert_eq!(df.column("pair.field_0").unwrap().get(0).unwrap(), AnyValue::String("hello"));
    assert_eq!(df.column("pair.field_1").unwrap().get(0).unwrap(), AnyValue::Int32(42));
    assert_eq!(df.column("triple.field_2").unwrap().get(0).unwrap(), AnyValue::Boolean(true));
    assert_eq!(df.column("single.field_0").unwrap().get(0).unwrap(), AnyValue::Int32(7));

    // Schema
    let schema = BareTuple::schema().unwrap();
    assert_eq!(schema.len(), 6);
    let empty = BareTuple::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 6));

    // 2. Option
    let opt_v = vec![
        OptTuple { pair: Some(("a".to_string(), 1)) },
        OptTuple { pair: None },
    ];
    let opt_df = opt_v.as_slice().to_dataframe().unwrap();
    assert_eq!(opt_df.shape(), (2, 2));
    assert_eq!(opt_df.column("pair.field_0").unwrap().get(0).unwrap(), AnyValue::String("a"));
    assert_eq!(opt_df.column("pair.field_0").unwrap().get(1).unwrap(), AnyValue::Null);
    assert_eq!(opt_df.column("pair.field_1").unwrap().get(0).unwrap(), AnyValue::Int32(1));
    assert_eq!(opt_df.column("pair.field_1").unwrap().get(1).unwrap(), AnyValue::Null);

    // 3. Vec
    let vec_v = VecTuple {
        pairs: vec![("x".to_string(), 10), ("y".to_string(), 20)],
    };
    let vec_df = vec_v.to_dataframe().unwrap();
    assert!(matches!(vec_df.column("pairs.field_0").unwrap().dtype(), DataType::List(_)));
    assert!(matches!(vec_df.column("pairs.field_1").unwrap().dtype(), DataType::List(_)));

    // 4. Vec<Option<tuple>>
    let vot_v = VecOptTuple {
        items: vec![
            Some(("a".to_string(), 1)),
            None,
            Some(("b".to_string(), 2)),
        ],
    };
    let _vot_df = vot_v.to_dataframe().unwrap();

    // 5. Option<Vec<tuple>>
    let ovt_v = vec![
        OptVecTuple {
            pairs: Some(vec![("a".to_string(), 1), ("b".to_string(), 2)]),
        },
        OptVecTuple { pairs: None },
    ];
    let _ovt_df = ovt_v.as_slice().to_dataframe().unwrap();

    // 6. Tuple in tuple struct
    let wt = WithTuple(99, ("hi".to_string(), false));
    let wt_df = wt.to_dataframe().unwrap();
    assert_eq!(
        wt_df.get_column_names(),
        ["field_0", "field_1.field_0", "field_1.field_1"]
    );
    assert_eq!(wt_df.column("field_0").unwrap().get(0).unwrap(), AnyValue::Int32(99));

    // 7. Smart pointer in element
    let sp = SmartPtrTuple {
        pair: (Box::new(42), "wrapped".to_string()),
    };
    let sp_df = sp.to_dataframe().unwrap();
    assert_eq!(sp_df.column("pair.field_0").unwrap().get(0).unwrap(), AnyValue::Int32(42));
    assert_eq!(sp_df.column("pair.field_1").unwrap().get(0).unwrap(), AnyValue::String("wrapped"));

    // 8. Vec<(Arc<u64>, String)>
    let vsp = VecSmartPtrTuple {
        items: vec![(Arc::new(100), "a".to_string()), (Arc::new(200), "b".to_string())],
    };
    let _vsp_df = vsp.to_dataframe().unwrap();

    // 9. Temporal
    let tt = TemporalTuple {
        times: (
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveTime::from_hms_opt(12, 0, 0).unwrap(),
        ),
        durs: vec![(chrono::Duration::seconds(60), "minute".to_string())],
    };
    let tt_df = tt.to_dataframe().unwrap();
    assert!(matches!(tt_df.column("times.field_0").unwrap().dtype(), DataType::Date));
    assert!(matches!(tt_df.column("times.field_1").unwrap().dtype(), DataType::Time));

    // 10. Nested tuples
    let nt = NestedTuple { nested: ((42, "inner".to_string()), true) };
    let nt_df = nt.to_dataframe().unwrap();
    let nt_cols = nt_df.get_column_names();
    assert_eq!(nt_cols, ["nested.field_0.field_0", "nested.field_0.field_1", "nested.field_1"]);
    assert_eq!(nt_df.column("nested.field_0.field_0").unwrap().get(0).unwrap(), AnyValue::Int32(42));
    assert_eq!(nt_df.column("nested.field_0.field_1").unwrap().get(0).unwrap(), AnyValue::String("inner"));
    assert_eq!(nt_df.column("nested.field_1").unwrap().get(0).unwrap(), AnyValue::Boolean(true));

    // 11. HashMap-rejection hint regression: Vec<(K, V)> compiles.
    let cm = ConvertedHashMap {
        metadata: vec![("key".to_string(), "value".to_string())],
    };
    let cm_df = cm.to_dataframe().unwrap();
    assert_eq!(cm_df.get_column_names(), ["metadata.field_0", "metadata.field_1"]);

    // 12. Tuple containing a nested struct
    let wi = WithInnerTuple {
        pair: (Inner { a: 7, b: 3.14 }, 99),
    };
    let wi_df = wi.to_dataframe().unwrap();
    let wi_cols: Vec<&str> = wi_df.get_column_names().iter().map(|n| n.as_str()).collect();
    assert!(wi_cols.contains(&"pair.field_0.a"));
    assert!(wi_cols.contains(&"pair.field_0.b"));
    assert!(wi_cols.contains(&"pair.field_1"));
    assert_eq!(wi_df.column("pair.field_0.a").unwrap().get(0).unwrap(), AnyValue::Int32(7));
    assert_eq!(wi_df.column("pair.field_1").unwrap().get(0).unwrap(), AnyValue::Int32(99));

    // 13. Vec of tuple containing a nested struct
    let vi = VecInnerTuple {
        items: vec![(Inner { a: 1, b: 1.0 }, 10), (Inner { a: 2, b: 2.0 }, 20)],
    };
    let vi_df = vi.to_dataframe().unwrap();
    let vi_cols: Vec<&str> = vi_df.get_column_names().iter().map(|n| n.as_str()).collect();
    assert!(vi_cols.contains(&"items.field_0.a"));
    assert!(vi_cols.contains(&"items.field_0.b"));
    assert!(vi_cols.contains(&"items.field_1"));
    assert!(matches!(vi_df.column("items.field_1").unwrap().dtype(), DataType::List(_)));

    // 14. Box<tuple>
    let bt = BoxTuple { pair: Box::new(("boxed".to_string(), 7)) };
    let bt_df = bt.to_dataframe().unwrap();
    assert_eq!(bt_df.column("pair.field_0").unwrap().get(0).unwrap(), AnyValue::String("boxed"));
    assert_eq!(bt_df.column("pair.field_1").unwrap().get(0).unwrap(), AnyValue::Int32(7));

    // Batch round-trip
    let batch = vec![
        BareTuple { pair: ("a".to_string(), 1), triple: (1.0, 2.0, true), single: (10,) },
        BareTuple { pair: ("b".to_string(), 2), triple: (3.0, 4.0, false), single: (20,) },
    ];
    let batch_df = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.shape(), (2, 6));
    assert_eq!(batch_df.column("pair.field_1").unwrap().get(1).unwrap(), AnyValue::Int32(2));
    assert_eq!(batch_df.column("single.field_0").unwrap().get(0).unwrap(), AnyValue::Int32(10));

    println!("Tuple field tests passed");
}
