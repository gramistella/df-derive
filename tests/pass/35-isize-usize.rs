// `isize`/`usize` coverage — Polars represents these via the widened
// `Int64`/`UInt64` lanes (no native platform-sized integer dtype). The
// encoder applies an `as i64` / `as u64` cast at every push site so the
// downstream chunked-array build matches the schema dtype directly. This
// test pins:
//
// - bare `isize` / `usize` (schema `Int64` / `UInt64`)
// - `Option<isize>` / `Option<usize>` (same dtype, validity bitmap)
// - `Vec<isize>` / `Vec<Option<isize>>` (schema `List<Int64>`)
// - `Vec<Vec<isize>>` (schema `List<List<Int64>>`)

use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame)]
struct IsizeUsizeBare {
    a: isize,
    b: usize,
    c: Option<isize>,
    d: Option<usize>,
}

#[derive(ToDataFrame)]
struct IsizeUsizeVecs {
    e: Vec<isize>,
    f: Vec<Option<isize>>,
    g: Vec<Vec<isize>>,
    h: Vec<usize>,
}

fn main() {
    // --- bare and Option shapes ---
    let bare_schema = IsizeUsizeBare::schema().unwrap();
    assert_eq!(
        bare_schema,
        vec![
            ("a".to_string(), DataType::Int64),
            ("b".to_string(), DataType::UInt64),
            ("c".to_string(), DataType::Int64),
            ("d".to_string(), DataType::UInt64),
        ]
    );

    let bare_items = vec![
        IsizeUsizeBare {
            a: -7,
            b: 42,
            c: Some(-99),
            d: Some(123),
        },
        IsizeUsizeBare {
            a: isize::MIN,
            b: usize::MAX,
            c: None,
            d: None,
        },
    ];
    let bare_df = bare_items.as_slice().to_dataframe().unwrap();
    assert_eq!(bare_df.shape(), (2, 4));
    assert_eq!(bare_df.column("a").unwrap().dtype(), &DataType::Int64);
    assert_eq!(bare_df.column("b").unwrap().dtype(), &DataType::UInt64);
    assert_eq!(bare_df.column("c").unwrap().dtype(), &DataType::Int64);
    assert_eq!(bare_df.column("d").unwrap().dtype(), &DataType::UInt64);

    assert_eq!(
        bare_df.column("a").unwrap().get(0).unwrap(),
        AnyValue::Int64(-7)
    );
    assert_eq!(
        bare_df.column("b").unwrap().get(0).unwrap(),
        AnyValue::UInt64(42)
    );
    assert_eq!(
        bare_df.column("c").unwrap().get(0).unwrap(),
        AnyValue::Int64(-99)
    );
    assert_eq!(
        bare_df.column("d").unwrap().get(0).unwrap(),
        AnyValue::UInt64(123)
    );

    assert_eq!(
        bare_df.column("a").unwrap().get(1).unwrap(),
        AnyValue::Int64(isize::MIN as i64)
    );
    assert_eq!(
        bare_df.column("b").unwrap().get(1).unwrap(),
        AnyValue::UInt64(usize::MAX as u64)
    );
    assert!(matches!(
        bare_df.column("c").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));
    assert!(matches!(
        bare_df.column("d").unwrap().get(1).unwrap(),
        AnyValue::Null
    ));

    // --- Vec-bearing shapes ---
    let vec_schema = IsizeUsizeVecs::schema().unwrap();
    let int64_list = DataType::List(Box::new(DataType::Int64));
    let uint64_list = DataType::List(Box::new(DataType::UInt64));
    let int64_list2 = DataType::List(Box::new(DataType::List(Box::new(DataType::Int64))));
    assert_eq!(
        vec_schema,
        vec![
            ("e".to_string(), int64_list.clone()),
            ("f".to_string(), int64_list.clone()),
            ("g".to_string(), int64_list2),
            ("h".to_string(), uint64_list),
        ]
    );

    let vec_items = vec![
        IsizeUsizeVecs {
            e: vec![1, 2, 3],
            f: vec![Some(10), None, Some(20)],
            g: vec![vec![1, 2], vec![3]],
            h: vec![100, 200],
        },
        IsizeUsizeVecs {
            e: vec![],
            f: vec![],
            g: vec![],
            h: vec![],
        },
    ];
    let vec_df = vec_items.as_slice().to_dataframe().unwrap();
    assert_eq!(vec_df.shape(), (2, 4));
    assert_eq!(
        vec_df.column("e").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Int64))
    );
    assert_eq!(
        vec_df.column("h").unwrap().dtype(),
        &DataType::List(Box::new(DataType::UInt64))
    );
    // Vec<Vec<isize>> runtime dtype carries the deeper List layer regardless
    // of schema (the schema flattens nested Vecs to one level by design).
    let g_runtime = vec_df.column("g").unwrap().dtype().clone();
    assert_eq!(
        g_runtime,
        DataType::List(Box::new(DataType::List(Box::new(DataType::Int64))))
    );

    // Row 0 sanity check on the inner-Option case.
    let f_row0 = vec_df.column("f").unwrap().get(0).unwrap();
    if let AnyValue::List(s) = f_row0 {
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(0).unwrap(), AnyValue::Int64(10));
        assert!(matches!(s.get(1).unwrap(), AnyValue::Null));
        assert_eq!(s.get(2).unwrap(), AnyValue::Int64(20));
    } else {
        panic!("expected List for f row 0, got {f_row0:?}");
    }

    // Empty slice yields an empty DataFrame with the declared schema.
    let empty_bare = <[IsizeUsizeBare] as ToDataFrameVec>::to_dataframe(&[]).unwrap();
    assert_eq!(empty_bare.shape(), (0, 4));
    let empty_vecs = <[IsizeUsizeVecs] as ToDataFrameVec>::to_dataframe(&[]).unwrap();
    assert_eq!(empty_vecs.shape(), (0, 4));
}
