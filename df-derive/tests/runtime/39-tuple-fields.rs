use crate::core::dataframe::{Decimal128Encode, ToDataFrame, ToDataFrameVec};
use chrono::{NaiveDate, NaiveTime};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::sync::Arc;

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

// 15. Tuple element with its own Vec wrapper under a Vec parent.
#[derive(ToDataFrame, Clone)]
struct VecTupleWithVecElement {
    items: Vec<(Vec<i32>, String)>,
}

// 16. Parent Vec element is optional, and the tuple element has its own Vec.
#[derive(ToDataFrame, Clone)]
struct VecOptTupleWithVecElement {
    items: Vec<Option<(Vec<i32>, String)>>,
}

// 17. Tuple element carries Option<Vec<_>> under a Vec parent.
#[derive(ToDataFrame, Clone)]
struct VecTupleWithOptVecElement {
    items: Vec<(Option<Vec<i32>>, String)>,
}

// 18. Optional tuple whose elements have non-Copy wrapper stacks.
#[derive(ToDataFrame, Clone)]
#[allow(clippy::type_complexity)]
struct OptTupleWithWrappedElements {
    pair: Option<(Vec<i32>, Option<Box<i32>>, Vec<Box<Option<i32>>>)>,
}

// 19. Parent tuple has interleaved Option / smart pointer / Option wrappers.
#[derive(ToDataFrame, Clone)]
struct OptionBoxOptionTuple {
    x: Option<Box<Option<(i32, String)>>>,
}

// 20. Parent tuple item has an Option and smart pointer above the tuple, and
// the projected element has its own Vec layer.
#[derive(ToDataFrame, Clone)]
#[allow(clippy::type_complexity)]
struct VecOptBoxTupleWithVecElement {
    items: Vec<Option<Box<(Vec<i32>, String)>>>,
}

// 21. Parent tuple is required, while a projected primitive element is optional.
#[derive(ToDataFrame, Clone)]
struct VecTupleWithOptionalElement {
    xs: Vec<(Option<i32>, String)>,
}

// 22. Borrowed string tuple elements lower to LeafSpec::AsStr without attrs.
#[derive(ToDataFrame, Clone)]
struct BorrowedStrTupleVec<'a> {
    xs: Vec<(&'a str, i32)>,
}

// 23. Parent Option tuple with borrowed Copy element.
#[derive(ToDataFrame, Clone)]
struct OptTupleWithBorrowedCopy<'a> {
    pair: Option<(&'a i32, bool)>,
}

// 24. Parent Option tuple with boxed Copy element.
#[derive(ToDataFrame, Clone)]
struct OptTupleWithBoxedCopy {
    pair: Option<(Box<i32>,)>,
}

// 25. Parent Option tuple with boxed non-Copy element.
#[derive(ToDataFrame, Clone)]
#[allow(clippy::box_collection)]
struct OptTupleWithBoxedString {
    pair: Option<(Box<String>,)>,
}

// 26. Parent Option tuple whose projected elements are independently optional.
#[derive(ToDataFrame, Clone)]
#[allow(clippy::type_complexity)]
struct OptTupleWithOptionalElements {
    pair: Option<(Option<String>, Option<Box<i32>>)>,
}

#[derive(ToDataFrame, Clone)]
struct Nested {
    id: i32,
    name: String,
}

// 27. Parent Vec<Option<tuple>> plus nested element Vec<Option<Box<Nested>>>
// and sibling Option<Vec<String>>.
#[derive(ToDataFrame, Clone)]
#[allow(clippy::type_complexity)]
struct VecOptTupleWithNestedVecAndOptVec {
    items: Vec<Option<(Vec<Option<Box<Nested>>>, Option<Vec<String>>)>>,
}

// 28. Parent Vec<Box<Option<tuple>>> plus nested Vec<Nested> and boxed
// optional scalar element.
#[derive(ToDataFrame, Clone)]
#[allow(clippy::type_complexity)]
struct VecBoxOptTupleWithNestedVecAndBoxedOpt {
    items: Vec<Box<Option<(Vec<Nested>, Box<Option<i64>>)>>>,
}

// 29. Box<((i32, String), Arc<bool>)>: boxed parent tuple, unwrapped nested
// tuple, and smart pointer projected from the second element.
#[derive(ToDataFrame, Clone)]
struct BoxNestedTupleWithArc {
    nested: Box<((i32, String), Arc<bool>)>,
}

// 30. Parent Option tuple with an implicit rust_decimal::Decimal leaf.
#[derive(ToDataFrame, Clone)]
struct OptTupleWithRustDecimal {
    pair: Option<(rust_decimal::Decimal,)>,
}

#[derive(Clone)]
struct MyDecimal(i128);

impl Decimal128Encode for MyDecimal {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        Some(self.0 + i128::from(target_scale))
    }
}

mod custom_decimal_tuple_backend {
    use super::*;

    type Decimal = MyDecimal;

    // 31. Parent Option tuple with a custom backend routed through the
    // default Decimal128Encode trait. Tuple elements cannot carry field-level
    // decimal attrs, so the syntactic `Decimal` alias exercises the implicit
    // decimal backend path while the impl lives on the custom type.
    #[derive(ToDataFrame, Clone)]
    pub(super) struct OptTupleWithCustomDecimal {
        pair: Option<(Decimal,)>,
    }

    pub(super) fn row(mantissa: Option<i128>) -> OptTupleWithCustomDecimal {
        OptTupleWithCustomDecimal {
            pair: mantissa.map(|value| (MyDecimal(value),)),
        }
    }
}

fn list_strings(value: AnyValue<'_>) -> Vec<Option<String>> {
    match value {
        AnyValue::List(series) => series
            .str()
            .unwrap()
            .into_iter()
            .map(|value| value.map(str::to_owned))
            .collect(),
        other => panic!("expected string List, got {other:?}"),
    }
}

fn list_i32s(value: AnyValue<'_>) -> Vec<Option<i32>> {
    match value {
        AnyValue::List(series) => series.i32().unwrap().into_iter().collect(),
        other => panic!("expected i32 List, got {other:?}"),
    }
}

fn list_i64s(value: AnyValue<'_>) -> Vec<Option<i64>> {
    match value {
        AnyValue::List(series) => series.i64().unwrap().into_iter().collect(),
        other => panic!("expected i64 List, got {other:?}"),
    }
}

fn nested_i32_lists(value: AnyValue<'_>) -> Vec<Option<Vec<Option<i32>>>> {
    let AnyValue::List(outer) = value else {
        panic!("expected outer List, got {value:?}");
    };
    (0..outer.len())
        .map(|idx| match outer.get(idx).unwrap() {
            AnyValue::List(inner) => Some(inner.i32().unwrap().into_iter().collect()),
            AnyValue::Null => None,
            other => panic!("expected inner i32 List or Null, got {other:?}"),
        })
        .collect()
}

fn nested_string_lists(value: AnyValue<'_>) -> Vec<Option<Vec<Option<String>>>> {
    let AnyValue::List(outer) = value else {
        panic!("expected outer List, got {value:?}");
    };
    (0..outer.len())
        .map(|idx| match outer.get(idx).unwrap() {
            AnyValue::List(inner) => Some(
                inner
                    .str()
                    .unwrap()
                    .into_iter()
                    .map(|value| value.map(str::to_owned))
                    .collect(),
            ),
            AnyValue::Null => None,
            other => panic!("expected inner string List or Null, got {other:?}"),
        })
        .collect()
}

fn decimal_mantissa(value: AnyValue<'_>) -> Option<i128> {
    match value {
        AnyValue::Decimal(value, _precision, _scale) => Some(value),
        AnyValue::Null => None,
        other => panic!("expected decimal or null, got {other:?}"),
    }
}

#[test]
fn runtime_semantics() {
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
    assert_eq!(
        df.column("pair.field_0").unwrap().get(0).unwrap(),
        AnyValue::String("hello")
    );
    assert_eq!(
        df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::Int32(42)
    );
    assert_eq!(
        df.column("triple.field_2").unwrap().get(0).unwrap(),
        AnyValue::Boolean(true)
    );
    assert_eq!(
        df.column("single.field_0").unwrap().get(0).unwrap(),
        AnyValue::Int32(7)
    );

    // Schema
    let schema = BareTuple::schema().unwrap();
    assert_eq!(schema.len(), 6);
    let empty = BareTuple::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 6));

    // 2. Option
    let opt_v = vec![
        OptTuple {
            pair: Some(("a".to_string(), 1)),
        },
        OptTuple { pair: None },
    ];
    let opt_df = opt_v.as_slice().to_dataframe().unwrap();
    assert_eq!(opt_df.shape(), (2, 2));
    assert_eq!(
        opt_df.column("pair.field_0").unwrap().get(0).unwrap(),
        AnyValue::String("a")
    );
    assert_eq!(
        opt_df.column("pair.field_0").unwrap().get(1).unwrap(),
        AnyValue::Null
    );
    assert_eq!(
        opt_df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::Int32(1)
    );
    assert_eq!(
        opt_df.column("pair.field_1").unwrap().get(1).unwrap(),
        AnyValue::Null
    );

    // 3. Vec
    let vec_v = VecTuple {
        pairs: vec![("x".to_string(), 10), ("y".to_string(), 20)],
    };
    let vec_df = vec_v.to_dataframe().unwrap();
    assert!(matches!(
        vec_df.column("pairs.field_0").unwrap().dtype(),
        DataType::List(_)
    ));
    assert!(matches!(
        vec_df.column("pairs.field_1").unwrap().dtype(),
        DataType::List(_)
    ));

    // 4. Vec<Option<tuple>>
    let vot_v = VecOptTuple {
        items: vec![Some(("a".to_string(), 1)), None, Some(("b".to_string(), 2))],
    };
    let vot_df = vot_v.to_dataframe().unwrap();
    assert_eq!(
        list_strings(vot_df.column("items.field_0").unwrap().get(0).unwrap()),
        vec![Some("a".to_string()), None, Some("b".to_string())],
    );
    assert_eq!(
        list_i32s(vot_df.column("items.field_1").unwrap().get(0).unwrap()),
        vec![Some(1), None, Some(2)],
    );

    // 5. Option<Vec<tuple>>
    let ovt_v = vec![
        OptVecTuple {
            pairs: Some(vec![("a".to_string(), 1), ("b".to_string(), 2)]),
        },
        OptVecTuple { pairs: None },
    ];
    let ovt_df = ovt_v.as_slice().to_dataframe().unwrap();
    assert_eq!(
        list_strings(ovt_df.column("pairs.field_0").unwrap().get(0).unwrap()),
        vec![Some("a".to_string()), Some("b".to_string())],
    );
    assert_eq!(
        list_i32s(ovt_df.column("pairs.field_1").unwrap().get(0).unwrap()),
        vec![Some(1), Some(2)],
    );
    assert_eq!(
        ovt_df.column("pairs.field_0").unwrap().get(1).unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        ovt_df.column("pairs.field_1").unwrap().get(1).unwrap(),
        AnyValue::Null,
    );

    // 6. Tuple in tuple struct
    let wt = WithTuple(99, ("hi".to_string(), false));
    let wt_df = wt.to_dataframe().unwrap();
    assert_eq!(
        wt_df.get_column_names(),
        ["field_0", "field_1.field_0", "field_1.field_1"]
    );
    assert_eq!(
        wt_df.column("field_0").unwrap().get(0).unwrap(),
        AnyValue::Int32(99)
    );

    // 7. Smart pointer in element
    let sp = SmartPtrTuple {
        pair: (Box::new(42), "wrapped".to_string()),
    };
    let sp_df = sp.to_dataframe().unwrap();
    assert_eq!(
        sp_df.column("pair.field_0").unwrap().get(0).unwrap(),
        AnyValue::Int32(42)
    );
    assert_eq!(
        sp_df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::String("wrapped")
    );

    // 8. Vec<(Arc<u64>, String)>
    let vsp = VecSmartPtrTuple {
        items: vec![
            (Arc::new(100), "a".to_string()),
            (Arc::new(200), "b".to_string()),
        ],
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
    assert!(matches!(
        tt_df.column("times.field_0").unwrap().dtype(),
        DataType::Date
    ));
    assert!(matches!(
        tt_df.column("times.field_1").unwrap().dtype(),
        DataType::Time
    ));

    // 10. Nested tuples
    let nt = NestedTuple {
        nested: ((42, "inner".to_string()), true),
    };
    let nt_df = nt.to_dataframe().unwrap();
    let nt_cols = nt_df.get_column_names();
    assert_eq!(
        nt_cols,
        [
            "nested.field_0.field_0",
            "nested.field_0.field_1",
            "nested.field_1"
        ]
    );
    assert_eq!(
        nt_df
            .column("nested.field_0.field_0")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(42)
    );
    assert_eq!(
        nt_df
            .column("nested.field_0.field_1")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::String("inner")
    );
    assert_eq!(
        nt_df.column("nested.field_1").unwrap().get(0).unwrap(),
        AnyValue::Boolean(true)
    );

    // 11. HashMap-rejection hint regression: Vec<(K, V)> compiles.
    let cm = ConvertedHashMap {
        metadata: vec![("key".to_string(), "value".to_string())],
    };
    let cm_df = cm.to_dataframe().unwrap();
    assert_eq!(
        cm_df.get_column_names(),
        ["metadata.field_0", "metadata.field_1"]
    );

    // 12. Tuple containing a nested struct
    let wi = WithInnerTuple {
        pair: (Inner { a: 7, b: 3.125 }, 99),
    };
    let wi_df = wi.to_dataframe().unwrap();
    let wi_cols: Vec<&str> = wi_df
        .get_column_names()
        .iter()
        .map(|n| n.as_str())
        .collect();
    assert!(wi_cols.contains(&"pair.field_0.a"));
    assert!(wi_cols.contains(&"pair.field_0.b"));
    assert!(wi_cols.contains(&"pair.field_1"));
    assert_eq!(
        wi_df.column("pair.field_0.a").unwrap().get(0).unwrap(),
        AnyValue::Int32(7)
    );
    assert_eq!(
        wi_df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::Int32(99)
    );

    // 13. Vec of tuple containing a nested struct
    let vi = VecInnerTuple {
        items: vec![(Inner { a: 1, b: 1.0 }, 10), (Inner { a: 2, b: 2.0 }, 20)],
    };
    let vi_df = vi.to_dataframe().unwrap();
    let vi_cols: Vec<&str> = vi_df
        .get_column_names()
        .iter()
        .map(|n| n.as_str())
        .collect();
    assert!(vi_cols.contains(&"items.field_0.a"));
    assert!(vi_cols.contains(&"items.field_0.b"));
    assert!(vi_cols.contains(&"items.field_1"));
    assert!(matches!(
        vi_df.column("items.field_1").unwrap().dtype(),
        DataType::List(_)
    ));

    // 14. Box<tuple>
    let bt = BoxTuple {
        pair: Box::new(("boxed".to_string(), 7)),
    };
    let bt_df = bt.to_dataframe().unwrap();
    assert_eq!(
        bt_df.column("pair.field_0").unwrap().get(0).unwrap(),
        AnyValue::String("boxed")
    );
    assert_eq!(
        bt_df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::Int32(7)
    );

    // 15. Vec<(Vec<i32>, String)>: projection happens between list layers.
    let vve = VecTupleWithVecElement {
        items: vec![
            (vec![1, 2], "a".to_string()),
            (vec![], "b".to_string()),
            (vec![3], "c".to_string()),
        ],
    };
    let vve_df = vve.to_dataframe().unwrap();
    assert!(matches!(
        vve_df.column("items.field_0").unwrap().dtype(),
        DataType::List(inner) if matches!(inner.as_ref(), DataType::List(_))
    ));
    assert_eq!(
        nested_i32_lists(vve_df.column("items.field_0").unwrap().get(0).unwrap()),
        vec![
            Some(vec![Some(1), Some(2)]),
            Some(Vec::<Option<i32>>::new()),
            Some(vec![Some(3)]),
        ],
    );
    assert_eq!(
        list_strings(vve_df.column("items.field_1").unwrap().get(0).unwrap()),
        vec![
            Some("a".to_string()),
            Some("b".to_string()),
            Some("c".to_string()),
        ],
    );

    // 16. Vec<Option<(Vec<i32>, String)>>: parent Option becomes inner-list validity.
    let vov = VecOptTupleWithVecElement {
        items: vec![
            Some((vec![10], "x".to_string())),
            None,
            Some((vec![20, 30], "z".to_string())),
        ],
    };
    let vov_df = vov.to_dataframe().unwrap();
    assert_eq!(
        nested_i32_lists(vov_df.column("items.field_0").unwrap().get(0).unwrap()),
        vec![Some(vec![Some(10)]), None, Some(vec![Some(20), Some(30)])],
    );

    // 17. Vec<(Option<Vec<i32>>, String)>: element Option becomes inner-list validity.
    let vov_elem = VecTupleWithOptVecElement {
        items: vec![
            (Some(vec![100]), "left".to_string()),
            (None, "middle".to_string()),
            (Some(vec![200, 300]), "right".to_string()),
        ],
    };
    let vov_elem_df = vov_elem.to_dataframe().unwrap();
    assert_eq!(
        nested_i32_lists(vov_elem_df.column("items.field_0").unwrap().get(0).unwrap()),
        vec![
            Some(vec![Some(100)]),
            None,
            Some(vec![Some(200), Some(300)])
        ],
    );

    // 18. Option<(Vec<i32>, Option<Box<i32>>, Vec<Box<Option<i32>>>)>:
    // parent Option must project tuple elements by reference when the
    // element wrapper is not Copy-projectable.
    let wrapped = vec![
        OptTupleWithWrappedElements {
            pair: Some((
                vec![1, 2],
                Some(Box::new(5)),
                vec![Box::new(Some(7)), Box::new(None)],
            )),
        },
        OptTupleWithWrappedElements { pair: None },
        OptTupleWithWrappedElements {
            pair: Some((vec![], None, vec![Box::new(None)])),
        },
    ];
    let wrapped_df = wrapped.as_slice().to_dataframe().unwrap();
    assert_eq!(
        list_i32s(wrapped_df.column("pair.field_0").unwrap().get(0).unwrap()),
        vec![Some(1), Some(2)],
    );
    assert_eq!(
        wrapped_df.column("pair.field_0").unwrap().get(1).unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        list_i32s(wrapped_df.column("pair.field_0").unwrap().get(2).unwrap()),
        Vec::<Option<i32>>::new(),
    );
    assert_eq!(
        wrapped_df.column("pair.field_1").unwrap().get(0).unwrap(),
        AnyValue::Int32(5),
    );
    assert_eq!(
        wrapped_df.column("pair.field_1").unwrap().get(1).unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        wrapped_df.column("pair.field_1").unwrap().get(2).unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        list_i32s(wrapped_df.column("pair.field_2").unwrap().get(0).unwrap()),
        vec![Some(7), None],
    );
    assert_eq!(
        wrapped_df.column("pair.field_2").unwrap().get(1).unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        list_i32s(wrapped_df.column("pair.field_2").unwrap().get(2).unwrap()),
        vec![None],
    );

    // 19. Option<Box<Option<(i32, String)>>>: collapse the real access chain,
    // not merely the number of Option layers.
    let interleaved = vec![
        OptionBoxOptionTuple {
            x: Some(Box::new(Some((11, "eleven".to_string())))),
        },
        OptionBoxOptionTuple {
            x: Some(Box::new(None)),
        },
        OptionBoxOptionTuple { x: None },
    ];
    let interleaved_df = interleaved.as_slice().to_dataframe().unwrap();
    assert_eq!(
        interleaved_df.column("x.field_0").unwrap().get(0).unwrap(),
        AnyValue::Int32(11),
    );
    assert_eq!(
        interleaved_df.column("x.field_1").unwrap().get(0).unwrap(),
        AnyValue::String("eleven"),
    );
    assert_eq!(
        interleaved_df.column("x.field_0").unwrap().get(1).unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        interleaved_df.column("x.field_1").unwrap().get(2).unwrap(),
        AnyValue::Null,
    );

    // 20. Vec<Option<Box<(Vec<i32>, String)>>>: projection consumes the
    // parent access chain once before the element's Vec layer is walked.
    let vec_interleaved = VecOptBoxTupleWithVecElement {
        items: vec![
            Some(Box::new((vec![1, 2], "left".to_string()))),
            None,
            Some(Box::new((vec![3], "right".to_string()))),
        ],
    };
    let vec_interleaved_df = vec_interleaved.to_dataframe().unwrap();
    assert_eq!(
        nested_i32_lists(
            vec_interleaved_df
                .column("items.field_0")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![Some(vec![Some(1), Some(2)]), None, Some(vec![Some(3)])],
    );
    assert_eq!(
        list_strings(
            vec_interleaved_df
                .column("items.field_1")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![Some("left".to_string()), None, Some("right".to_string())],
    );

    // 21. Vec<(Option<i32>, String)>: the Option belongs to field_0 only;
    // field_1 remains present for every tuple item.
    let optional_elem = VecTupleWithOptionalElement {
        xs: vec![
            (Some(1), "one".to_string()),
            (None, "missing".to_string()),
            (Some(3), "three".to_string()),
        ],
    };
    let optional_elem_df = optional_elem.to_dataframe().unwrap();
    assert_eq!(
        list_i32s(
            optional_elem_df
                .column("xs.field_0")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![Some(1), None, Some(3)],
    );
    assert_eq!(
        list_strings(
            optional_elem_df
                .column("xs.field_1")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![
            Some("one".to_string()),
            Some("missing".to_string()),
            Some("three".to_string()),
        ],
    );

    // 22. Vec<(&str, i32)>: borrowed string tuple elements use the default
    // AsStr lowering and should share the same list string emitter.
    let borrowed = BorrowedStrTupleVec {
        xs: vec![("alpha", 10), ("beta", 20)],
    };
    let borrowed_df = borrowed.to_dataframe().unwrap();
    assert_eq!(
        list_strings(borrowed_df.column("xs.field_0").unwrap().get(0).unwrap()),
        vec![Some("alpha".to_string()), Some("beta".to_string())],
    );
    assert_eq!(
        list_i32s(borrowed_df.column("xs.field_1").unwrap().get(0).unwrap()),
        vec![Some(10), Some(20)],
    );

    // 23. Option<(&i32, bool)>: parent Option projection must dereference the
    // borrowed numeric element before feeding the Copy Option leaf.
    let borrowed_scalar = 123;
    let borrowed_copy = vec![
        OptTupleWithBorrowedCopy {
            pair: Some((&borrowed_scalar, true)),
        },
        OptTupleWithBorrowedCopy { pair: None },
    ];
    let borrowed_copy_df = borrowed_copy.as_slice().to_dataframe().unwrap();
    assert_eq!(
        borrowed_copy_df
            .column("pair.field_0")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(123),
    );
    assert_eq!(
        borrowed_copy_df
            .column("pair.field_1")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Boolean(true),
    );
    assert_eq!(
        borrowed_copy_df
            .column("pair.field_0")
            .unwrap()
            .get(1)
            .unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        borrowed_copy_df
            .column("pair.field_1")
            .unwrap()
            .get(1)
            .unwrap(),
        AnyValue::Null,
    );

    // 24. Option<(Box<i32>,)>: projection must not move the Box out of the
    // borrowed tuple; it dereferences first and copies the i32.
    let boxed_copy = vec![
        OptTupleWithBoxedCopy {
            pair: Some((Box::new(456),)),
        },
        OptTupleWithBoxedCopy { pair: None },
    ];
    let boxed_copy_df = boxed_copy.as_slice().to_dataframe().unwrap();
    assert_eq!(
        boxed_copy_df
            .column("pair.field_0")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(456),
    );
    assert_eq!(
        boxed_copy_df
            .column("pair.field_0")
            .unwrap()
            .get(1)
            .unwrap(),
        AnyValue::Null,
    );

    // 25. Option<(Box<String>,)>: non-Copy smart-pointer elements project as
    // references after applying the element smart-pointer depth.
    let boxed_string = vec![
        OptTupleWithBoxedString {
            pair: Some((Box::new("boxed".to_string()),)),
        },
        OptTupleWithBoxedString { pair: None },
    ];
    let boxed_string_df = boxed_string.as_slice().to_dataframe().unwrap();
    assert_eq!(
        boxed_string_df
            .column("pair.field_0")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::String("boxed"),
    );
    assert_eq!(
        boxed_string_df
            .column("pair.field_0")
            .unwrap()
            .get(1)
            .unwrap(),
        AnyValue::Null,
    );

    // 26. Option<(Option<String>, Option<Box<i32>>)>
    let optional_elements = vec![
        OptTupleWithOptionalElements {
            pair: Some((Some("present".to_string()), Some(Box::new(5)))),
        },
        OptTupleWithOptionalElements {
            pair: Some((None, None)),
        },
        OptTupleWithOptionalElements { pair: None },
    ];
    let optional_elements_df = optional_elements.as_slice().to_dataframe().unwrap();
    assert_eq!(
        optional_elements_df
            .column("pair.field_0")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::String("present"),
    );
    assert_eq!(
        optional_elements_df
            .column("pair.field_1")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(5),
    );
    assert_eq!(
        optional_elements_df
            .column("pair.field_0")
            .unwrap()
            .get(1)
            .unwrap(),
        AnyValue::Null,
    );
    assert_eq!(
        optional_elements_df
            .column("pair.field_1")
            .unwrap()
            .get(2)
            .unwrap(),
        AnyValue::Null,
    );

    // 27. Vec<Option<(Vec<Option<Box<Nested>>>, Option<Vec<String>>)>>
    let nested_opt_vec = VecOptTupleWithNestedVecAndOptVec {
        items: vec![
            Some((
                vec![
                    Some(Box::new(Nested {
                        id: 1,
                        name: "one".to_string(),
                    })),
                    None,
                    Some(Box::new(Nested {
                        id: 2,
                        name: "two".to_string(),
                    })),
                ],
                Some(vec!["left".to_string(), "right".to_string()]),
            )),
            None,
            Some((Vec::new(), None)),
        ],
    };
    let nested_opt_vec_df = nested_opt_vec.to_dataframe().unwrap();
    assert_eq!(
        nested_i32_lists(
            nested_opt_vec_df
                .column("items.field_0.id")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![Some(vec![Some(1), None, Some(2)]), None, Some(Vec::new())],
    );
    assert_eq!(
        nested_string_lists(
            nested_opt_vec_df
                .column("items.field_0.name")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![
            Some(vec![Some("one".to_string()), None, Some("two".to_string())]),
            None,
            Some(Vec::new()),
        ],
    );
    assert_eq!(
        nested_string_lists(
            nested_opt_vec_df
                .column("items.field_1")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![
            Some(vec![Some("left".to_string()), Some("right".to_string())]),
            None,
            None,
        ],
    );

    // 28. Vec<Box<Option<(Vec<Nested>, Box<Option<i64>>)>>>
    let boxed_opt_nested = VecBoxOptTupleWithNestedVecAndBoxedOpt {
        items: vec![
            Box::new(Some((
                vec![
                    Nested {
                        id: 10,
                        name: "ten".to_string(),
                    },
                    Nested {
                        id: 11,
                        name: "eleven".to_string(),
                    },
                ],
                Box::new(Some(100)),
            ))),
            Box::new(None),
            Box::new(Some((Vec::new(), Box::new(None)))),
            Box::new(Some((
                vec![Nested {
                    id: 12,
                    name: "twelve".to_string(),
                }],
                Box::new(Some(120)),
            ))),
        ],
    };
    let boxed_opt_nested_df = boxed_opt_nested.to_dataframe().unwrap();
    assert_eq!(
        nested_i32_lists(
            boxed_opt_nested_df
                .column("items.field_0.id")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![
            Some(vec![Some(10), Some(11)]),
            None,
            Some(Vec::new()),
            Some(vec![Some(12)]),
        ],
    );
    assert_eq!(
        nested_string_lists(
            boxed_opt_nested_df
                .column("items.field_0.name")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![
            Some(vec![Some("ten".to_string()), Some("eleven".to_string())]),
            None,
            Some(Vec::new()),
            Some(vec![Some("twelve".to_string())]),
        ],
    );
    assert_eq!(
        list_i64s(
            boxed_opt_nested_df
                .column("items.field_1")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        vec![Some(100), None, None, Some(120)],
    );

    // 29. Box<((i32, String), Arc<bool>)>
    let boxed_nested_tuple = BoxNestedTupleWithArc {
        nested: Box::new(((8, "boxed nested".to_string()), Arc::new(true))),
    };
    let boxed_nested_tuple_df = boxed_nested_tuple.to_dataframe().unwrap();
    assert_eq!(
        boxed_nested_tuple_df.get_column_names(),
        [
            "nested.field_0.field_0",
            "nested.field_0.field_1",
            "nested.field_1"
        ],
    );
    assert_eq!(
        boxed_nested_tuple_df
            .column("nested.field_0.field_0")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(8),
    );
    assert_eq!(
        boxed_nested_tuple_df
            .column("nested.field_0.field_1")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::String("boxed nested"),
    );
    assert_eq!(
        boxed_nested_tuple_df
            .column("nested.field_1")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Boolean(true),
    );

    // 30. Option<(rust_decimal::Decimal,)>: parent Option projection yields
    // Option<&Decimal>; decimal UFCS dispatch must still reach the backend.
    let rust_decimal_rows = vec![
        OptTupleWithRustDecimal {
            pair: Some((rust_decimal::Decimal::new(123, 2),)),
        },
        OptTupleWithRustDecimal { pair: None },
    ];
    let rust_decimal_df = rust_decimal_rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        rust_decimal_df.column("pair.field_0").unwrap().dtype(),
        &DataType::Decimal(38, 10),
    );
    assert_eq!(
        decimal_mantissa(
            rust_decimal_df
                .column("pair.field_0")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        Some(12_300_000_000),
    );
    assert_eq!(
        decimal_mantissa(
            rust_decimal_df
                .column("pair.field_0")
                .unwrap()
                .get(1)
                .unwrap()
        ),
        None,
    );

    // 31. Option<(Decimal alias to MyDecimal,)>: custom backend dispatch
    // uses the same projected Option<&T> shape as rust_decimal.
    let custom_decimal_rows = vec![
        custom_decimal_tuple_backend::row(Some(700)),
        custom_decimal_tuple_backend::row(None),
    ];
    let custom_decimal_df = custom_decimal_rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        custom_decimal_df.column("pair.field_0").unwrap().dtype(),
        &DataType::Decimal(38, 10),
    );
    assert_eq!(
        decimal_mantissa(
            custom_decimal_df
                .column("pair.field_0")
                .unwrap()
                .get(0)
                .unwrap()
        ),
        Some(710),
    );
    assert_eq!(
        decimal_mantissa(
            custom_decimal_df
                .column("pair.field_0")
                .unwrap()
                .get(1)
                .unwrap()
        ),
        None,
    );

    // Batch round-trip
    let batch = vec![
        BareTuple {
            pair: ("a".to_string(), 1),
            triple: (1.0, 2.0, true),
            single: (10,),
        },
        BareTuple {
            pair: ("b".to_string(), 2),
            triple: (3.0, 4.0, false),
            single: (20,),
        },
    ];
    let batch_df = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.shape(), (2, 6));
    assert_eq!(
        batch_df.column("pair.field_1").unwrap().get(1).unwrap(),
        AnyValue::Int32(2)
    );
    assert_eq!(
        batch_df.column("single.field_0").unwrap().get(0).unwrap(),
        AnyValue::Int32(10)
    );

    println!("Tuple field tests passed");
}
