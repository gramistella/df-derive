// Regression: a `[Vec, Option, Vec, ...]` wrapper stack must preserve the
// middle `Option` as list-level validity on the inner `Vec`. Earlier
// `normalize_wrappers` discarded any `Option` accumulated between two `Vec`
// wraps — `Vec<Option<Vec<T>>>` produced shapes equivalent to
// `Vec<Vec<T>>`, dropping the inner-list validity bit and silently
// mistyping the leaf push site (the `Option::iter()` flatten in the
// generated populator yields the inner element type directly, so a
// dropped option layer fed `Option<T>` into a slot expecting `T`).
//
// The shapes below all carry at least one mid-stack `Option`. Each row's
// inner `Vec` is independently nullable; where the schema is
// `List<List<Int32>>` Polars' single-validity-bit-per-level rule means
// `Some(None)` and `None` collapse to the same null cell (the test
// observes them via the `AnyValue::Null` cell and compares lengths and
// values for the populated cases).

use crate::core::dataframe::ToDataFrame;
use df_derive::ToDataFrame;
use polars::prelude::*;
use pretty_assertions::assert_eq;

#[derive(ToDataFrame)]
struct MidStackOptions {
    id: i32,
    // [Vec, Option, Vec] — middle Option is the inner-list-level validity.
    voi32: Vec<Option<Vec<i32>>>,
    // [Vec, Option, Vec, Option] — middle Option is the inner-list-level
    // validity, inner Option is per-element validity at the leaf.
    voos: Vec<Option<Vec<Option<String>>>>,
    // [Option, Vec, Option, Vec] — outer Option folds into outer-list-level
    // validity, middle Option is inner-list-level validity.
    ovoi: Option<Vec<Option<Vec<i32>>>>,
}

fn list_list_i32() -> DataType {
    DataType::List(Box::new(DataType::List(Box::new(DataType::Int32))))
}

fn list_list_string() -> DataType {
    DataType::List(Box::new(DataType::List(Box::new(DataType::String))))
}

#[test]
fn runtime_semantics() {
    let row = MidStackOptions {
        id: 1,
        voi32: vec![Some(vec![1, 2, 3]), None, Some(vec![]), Some(vec![10, 20])],
        voos: vec![
            Some(vec![Some("a".to_string()), None, Some("c".to_string())]),
            None,
            Some(vec![]),
        ],
        ovoi: Some(vec![Some(vec![100, 200]), None, Some(vec![300])]),
    };
    let df = row.to_dataframe().unwrap();

    // Schema inspection: each runtime column must carry the full nesting
    // depth (one `List<>` per `Vec` wrapper). The pre-fix bug collapsed
    // `Vec<Option<Vec<T>>>` schemas to `List<T>`, dropping a level.
    let schema = df.schema();
    assert_eq!(schema.get("voi32").unwrap(), &list_list_i32());
    assert_eq!(schema.get("voos").unwrap(), &list_list_string());
    assert_eq!(schema.get("ovoi").unwrap(), &list_list_i32());

    // voi32: outer list of 4 inner lists; element 1 is null, element 2 is
    // present-but-empty, others populated.
    let voi32_av = df.column("voi32").unwrap().get(0).unwrap();
    let AnyValue::List(voi32_outer) = voi32_av else {
        panic!("voi32 must be List, got {voi32_av:?}");
    };
    assert_eq!(voi32_outer.len(), 4);
    assert_eq!(
        voi32_outer.dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );

    let inner_0 = match voi32_outer.get(0).unwrap() {
        AnyValue::List(s) => s.i32().unwrap().into_iter().collect::<Vec<_>>(),
        v => panic!("voi32[0] must be List, got {v:?}"),
    };
    assert_eq!(inner_0, vec![Some(1), Some(2), Some(3)]);

    let inner_1 = voi32_outer.get(1).unwrap();
    assert_eq!(
        inner_1,
        AnyValue::Null,
        "voi32[1] must be Null (mid-stack Option)"
    );

    let inner_2 = match voi32_outer.get(2).unwrap() {
        AnyValue::List(s) => s.i32().unwrap().into_iter().collect::<Vec<_>>(),
        v => panic!("voi32[2] must be List(empty), got {v:?}"),
    };
    assert_eq!(inner_2, Vec::<Option<i32>>::new());

    let inner_3 = match voi32_outer.get(3).unwrap() {
        AnyValue::List(s) => s.i32().unwrap().into_iter().collect::<Vec<_>>(),
        v => panic!("voi32[3] must be List, got {v:?}"),
    };
    assert_eq!(inner_3, vec![Some(10), Some(20)]);

    // voos: middle Option + leaf Option both carried.
    let voos_av = df.column("voos").unwrap().get(0).unwrap();
    let AnyValue::List(voos_outer) = voos_av else {
        panic!("voos must be List, got {voos_av:?}");
    };
    assert_eq!(voos_outer.len(), 3);
    assert_eq!(
        voos_outer.dtype(),
        &DataType::List(Box::new(DataType::String))
    );

    let voos_0 = match voos_outer.get(0).unwrap() {
        AnyValue::List(s) => s
            .str()
            .unwrap()
            .into_iter()
            .map(|o| o.map(str::to_string))
            .collect::<Vec<_>>(),
        v => panic!("voos[0] must be List, got {v:?}"),
    };
    assert_eq!(
        voos_0,
        vec![Some("a".to_string()), None, Some("c".to_string())],
    );
    assert_eq!(
        voos_outer.get(1).unwrap(),
        AnyValue::Null,
        "voos[1] must be Null (mid-stack Option)"
    );
    let voos_2 = match voos_outer.get(2).unwrap() {
        AnyValue::List(s) => s
            .str()
            .unwrap()
            .into_iter()
            .map(|o| o.map(str::to_string))
            .collect::<Vec<_>>(),
        v => panic!("voos[2] must be List(empty), got {v:?}"),
    };
    assert_eq!(voos_2, Vec::<Option<String>>::new());

    // ovoi: outer Option present, middle Option carries the inner-list
    // validity for the second element.
    let ovoi_av = df.column("ovoi").unwrap().get(0).unwrap();
    let AnyValue::List(ovoi_outer) = ovoi_av else {
        panic!("ovoi must be List, got {ovoi_av:?}");
    };
    assert_eq!(ovoi_outer.len(), 3);
    assert_eq!(
        ovoi_outer.dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );

    let ovoi_0 = match ovoi_outer.get(0).unwrap() {
        AnyValue::List(s) => s.i32().unwrap().into_iter().collect::<Vec<_>>(),
        v => panic!("ovoi[0] must be List, got {v:?}"),
    };
    assert_eq!(ovoi_0, vec![Some(100), Some(200)]);
    assert_eq!(
        ovoi_outer.get(1).unwrap(),
        AnyValue::Null,
        "ovoi[1] must be Null"
    );
    let ovoi_2 = match ovoi_outer.get(2).unwrap() {
        AnyValue::List(s) => s.i32().unwrap().into_iter().collect::<Vec<_>>(),
        v => panic!("ovoi[2] must be List, got {v:?}"),
    };
    assert_eq!(ovoi_2, vec![Some(300)]);
}
