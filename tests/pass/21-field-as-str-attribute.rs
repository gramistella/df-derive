use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(Clone, Debug, PartialEq)]
enum Status {
    Active,
    Inactive,
}

// AsRef<str> instead of Display: this is what `as_str` requires.
impl AsRef<str> for Status {
    fn as_ref(&self) -> &str {
        match self {
            Status::Active => "ACTIVE",
            Status::Inactive => "INACTIVE",
        }
    }
}

#[derive(ToDataFrame, Clone)]
struct WithEnums {
    #[df_derive(as_str)]
    status: Status,
    #[df_derive(as_str)]
    opt_status: Option<Status>,
    #[df_derive(as_str)]
    statuses: Vec<Status>,
    #[df_derive(as_str)]
    opt_statuses: Option<Vec<Status>>,
}

fn assert_col_str(df: &DataFrame, col: &str, expected: &str) {
    let v = df.column(col).unwrap().get(0).unwrap();
    match v {
        AnyValue::String(s) => assert_eq!(s, expected),
        AnyValue::StringOwned(ref s) => assert_eq!(s.as_str(), expected),
        other => panic!("unexpected AnyValue for {}: {:?}", col, other),
    }
}

fn assert_list_strs(df: &DataFrame, col: &str, expected: &[&str]) {
    let av = df.column(col).unwrap().get(0).unwrap();
    if let AnyValue::List(inner) = av {
        let vals: Vec<String> = inner
            .iter()
            .map(|v| match v {
                AnyValue::String(s) => s.to_string(),
                AnyValue::StringOwned(ref s) => s.to_string(),
                other => panic!("unexpected AnyValue in {}: {:?}", col, other),
            })
            .collect();
        let expected_owned: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(vals, expected_owned);
    } else {
        panic!("expected List for {}, got {:?}", col, av)
    }
}

fn main() {
    println!("--- Testing #[df_derive(as_str)] attribute for AsRef<str> serialization ---");

    let s = WithEnums {
        status: Status::Active,
        opt_status: Some(Status::Inactive),
        statuses: vec![Status::Active, Status::Inactive],
        opt_statuses: Some(vec![Status::Inactive, Status::Active]),
    };

    println!("🔄 Converting single value to DataFrame (row-wise path)...");
    let df = s.clone().to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 4));

    let schema = df.schema();
    assert_eq!(schema.get("status"), Some(&DataType::String));
    assert_eq!(schema.get("opt_status"), Some(&DataType::String));
    assert_eq!(
        schema.get("statuses"),
        Some(&DataType::List(Box::new(DataType::String)))
    );
    assert_eq!(
        schema.get("opt_statuses"),
        Some(&DataType::List(Box::new(DataType::String)))
    );

    assert_col_str(&df, "status", "ACTIVE");
    assert_col_str(&df, "opt_status", "INACTIVE");
    assert_list_strs(&df, "statuses", &["ACTIVE", "INACTIVE"]);
    assert_list_strs(&df, "opt_statuses", &["INACTIVE", "ACTIVE"]);

    println!("🔄 Converting Vec<WithEnums> to DataFrame (columnar path)...");
    let batch = vec![
        s.clone(),
        WithEnums {
            status: Status::Inactive,
            opt_status: None,
            statuses: vec![],
            opt_statuses: None,
        },
        s.clone(),
    ];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (3, 4));

    let status_col = df_batch.column("status").unwrap();
    let status_strs: Vec<Option<&str>> = status_col.str().unwrap().into_iter().collect();
    assert_eq!(
        status_strs,
        vec![Some("ACTIVE"), Some("INACTIVE"), Some("ACTIVE")]
    );

    let opt_status_col = df_batch.column("opt_status").unwrap();
    let opt_status_strs: Vec<Option<&str>> = opt_status_col.str().unwrap().into_iter().collect();
    assert_eq!(
        opt_status_strs,
        vec![Some("INACTIVE"), None, Some("INACTIVE")]
    );

    println!("🔄 Verifying Vec<Status> in batch...");
    let statuses_col = df_batch.column("statuses").unwrap();
    let row0 = statuses_col.get(0).unwrap();
    let row1 = statuses_col.get(1).unwrap();
    if let AnyValue::List(inner) = row0 {
        let vals: Vec<Option<&str>> = inner.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![Some("ACTIVE"), Some("INACTIVE")]);
    } else {
        panic!("expected list for statuses[0]");
    }
    if let AnyValue::List(inner) = row1 {
        let n: usize = inner.str().unwrap().into_iter().count();
        assert_eq!(n, 0);
    } else {
        panic!("expected list for statuses[1]");
    }

    println!("🔄 Verifying Option<Vec<Status>> in batch (Some, None, Some)...");
    let opt_statuses_col = df_batch.column("opt_statuses").unwrap();
    if let AnyValue::List(inner) = opt_statuses_col.get(0).unwrap() {
        let vals: Vec<Option<&str>> = inner.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![Some("INACTIVE"), Some("ACTIVE")]);
    } else {
        panic!("expected list for opt_statuses[0]");
    }
    let row1_opt = opt_statuses_col.get(1).unwrap();
    assert!(
        matches!(row1_opt, AnyValue::Null),
        "expected null for opt_statuses[1] (Option::None), got {:?}",
        row1_opt
    );
    if let AnyValue::List(inner) = opt_statuses_col.get(2).unwrap() {
        let vals: Vec<Option<&str>> = inner.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![Some("INACTIVE"), Some("ACTIVE")]);
    } else {
        panic!("expected list for opt_statuses[2]");
    }

    println!("🔄 Empty DataFrame schema check...");
    let empty = WithEnums::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 4));
    let schema = empty.schema();
    assert_eq!(schema.get("status"), Some(&DataType::String));
    assert_eq!(schema.get("opt_status"), Some(&DataType::String));
    assert_eq!(
        schema.get("statuses"),
        Some(&DataType::List(Box::new(DataType::String)))
    );
    assert_eq!(
        schema.get("opt_statuses"),
        Some(&DataType::List(Box::new(DataType::String)))
    );

    println!("\n🔄 Generic-leaf case: #[df_derive(as_str)] on a generic field type T...");
    test_generic_as_str();

    println!("\n🔄 Deep wrapper shapes: Vec<Vec<T>>, Vec<Option<T>>...");
    test_deep_wrappers();

    println!("\n✅ #[df_derive(as_str)] attribute test completed successfully!");
}

// Generic-leaf `as_str`: the macro injects `T: ToDataFrame + Columnar + Clone`
// on every type parameter (struct-level — see `impl_parts_with_bounds`), so a
// generic field with `as_str` requires `T` to satisfy those *plus* `AsRef<str>`
// from the const-fn assert. The test's `LabelStr` derives `ToDataFrame` to
// supply the framework bounds and impls `AsRef<str>` for the borrowing path.
#[derive(ToDataFrame, Clone)]
struct LabelStr {
    label: String,
}

impl AsRef<str> for LabelStr {
    fn as_ref(&self) -> &str {
        &self.label
    }
}

#[derive(ToDataFrame, Clone)]
struct GenericWrap<T>
where
    T: Clone + AsRef<str>,
{
    id: u32,
    #[df_derive(as_str)]
    label: T,
    #[df_derive(as_str)]
    opt_label: Option<T>,
    #[df_derive(as_str)]
    labels: Vec<T>,
}

fn test_generic_as_str() {
    let item = GenericWrap::<LabelStr> {
        id: 7,
        label: LabelStr {
            label: "ACTIVE".into(),
        },
        opt_label: Some(LabelStr {
            label: "INACTIVE".into(),
        }),
        labels: vec![
            LabelStr {
                label: "INACTIVE".into(),
            },
            LabelStr {
                label: "ACTIVE".into(),
            },
        ],
    };
    let df = item.to_dataframe().unwrap();
    let schema = df.schema();
    assert_eq!(schema.get("label"), Some(&DataType::String));
    assert_eq!(schema.get("opt_label"), Some(&DataType::String));
    assert_eq!(
        schema.get("labels"),
        Some(&DataType::List(Box::new(DataType::String)))
    );
    assert_col_str(&df, "label", "ACTIVE");
    assert_col_str(&df, "opt_label", "INACTIVE");
    assert_list_strs(&df, "labels", &["INACTIVE", "ACTIVE"]);

    // Columnar batch with a `None` row to exercise the `Option<T>` generic
    // path through the columnar populator.
    let batch = vec![
        item.clone(),
        GenericWrap::<LabelStr> {
            id: 8,
            label: LabelStr {
                label: "INACTIVE".into(),
            },
            opt_label: None,
            labels: vec![],
        },
        item.clone(),
    ];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (3, 4));
    let labels: Vec<Option<&str>> = df_batch
        .column("label")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(
        labels,
        vec![Some("ACTIVE"), Some("INACTIVE"), Some("ACTIVE")]
    );
    let opt_labels: Vec<Option<&str>> = df_batch
        .column("opt_label")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(opt_labels, vec![Some("INACTIVE"), None, Some("INACTIVE")]);

    let labels_col = df_batch.column("labels").unwrap();
    if let AnyValue::List(inner) = labels_col.get(0).unwrap() {
        let vals: Vec<Option<&str>> = inner.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![Some("INACTIVE"), Some("ACTIVE")]);
    } else {
        panic!("expected list for labels[0]");
    }
    if let AnyValue::List(inner) = labels_col.get(1).unwrap() {
        let n: usize = inner.str().unwrap().into_iter().count();
        assert_eq!(n, 0);
    } else {
        panic!("expected list for labels[1]");
    }
}

// `Vec<Vec<T>>+as_str` and `Vec<Option<T>>+as_str` exercise depth-2 wrapper
// shapes around the leaf. The `Vec<Vec<T>>` shape was previously broken by a
// dangling-reference bug (the inner-Series `Vec<&str>` borrowed from a
// temporary `(*elem).clone()` that dropped at the previous `;`). The
// regression check builds the DataFrame and asserts shape/values; if codegen
// regresses to the dangling pattern, this test fails to compile.
#[derive(ToDataFrame, Clone)]
struct DeepWrappers {
    #[df_derive(as_str)]
    deep: Vec<Vec<Status>>,
    #[df_derive(as_str)]
    vec_opt: Vec<Option<Status>>,
    // Depth-3 case: locks the recursive shadow behavior of
    // `__df_derive_elem_owned` across two levels of fallback nesting.
    #[df_derive(as_str)]
    triple: Vec<Vec<Vec<Status>>>,
}

fn test_deep_wrappers() {
    let item = DeepWrappers {
        deep: vec![
            vec![Status::Active, Status::Inactive],
            vec![Status::Inactive],
        ],
        vec_opt: vec![Some(Status::Active), None, Some(Status::Inactive)],
        triple: vec![
            vec![vec![Status::Active, Status::Inactive], vec![Status::Active]],
            vec![],
        ],
    };
    let df = item.clone().to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 3));

    // `triple` schema is `List<List<List<String>>>`.
    let triple_dtype = df.schema().get("triple").unwrap().clone();
    assert_eq!(
        triple_dtype,
        DataType::List(Box::new(DataType::List(Box::new(DataType::List(Box::new(
            DataType::String
        ))))))
    );
    let triple_col = df.column("triple").unwrap();
    if let AnyValue::List(outer) = triple_col.get(0).unwrap() {
        let mid_lists: Vec<_> = outer.list().unwrap().into_iter().collect();
        assert_eq!(mid_lists.len(), 2);
        // Mid-list 0 contains two inner lists.
        let mid0 = mid_lists[0].as_ref().unwrap();
        let inner_lists: Vec<_> = mid0.list().unwrap().into_iter().collect();
        assert_eq!(inner_lists.len(), 2);
        let inner0: Vec<Option<&str>> = inner_lists[0]
            .as_ref()
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(inner0, vec![Some("ACTIVE"), Some("INACTIVE")]);
        let inner1: Vec<Option<&str>> = inner_lists[1]
            .as_ref()
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(inner1, vec![Some("ACTIVE")]);
        // Mid-list 1 is empty.
        let mid1 = mid_lists[1].as_ref().unwrap();
        assert_eq!(mid1.list().unwrap().into_iter().count(), 0);
    } else {
        panic!("expected outer list for triple");
    }

    // `deep` should be List<List<String>>.
    let deep_dtype = df.schema().get("deep").unwrap().clone();
    assert_eq!(
        deep_dtype,
        DataType::List(Box::new(DataType::List(Box::new(DataType::String))))
    );
    let deep_col = df.column("deep").unwrap();
    if let AnyValue::List(outer) = deep_col.get(0).unwrap() {
        let lists: Vec<_> = outer.list().unwrap().into_iter().collect();
        assert_eq!(lists.len(), 2);
        let row0: Vec<Option<&str>> = lists[0].as_ref().unwrap().str().unwrap().into_iter().collect();
        assert_eq!(row0, vec![Some("ACTIVE"), Some("INACTIVE")]);
        let row1: Vec<Option<&str>> = lists[1].as_ref().unwrap().str().unwrap().into_iter().collect();
        assert_eq!(row1, vec![Some("INACTIVE")]);
    } else {
        panic!("expected outer list for deep");
    }

    // `vec_opt` is List<String> (Option<T> at the inner level becomes nullable str).
    assert_eq!(
        df.schema().get("vec_opt").unwrap(),
        &DataType::List(Box::new(DataType::String))
    );
    let vec_opt_col = df.column("vec_opt").unwrap();
    if let AnyValue::List(inner) = vec_opt_col.get(0).unwrap() {
        let vals: Vec<Option<&str>> = inner.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![Some("ACTIVE"), None, Some("INACTIVE")]);
    } else {
        panic!("expected list for vec_opt");
    }

    // Columnar batch path. Includes a fully-empty middle row plus a third
    // row with a different intermixed Some/None pattern so the columnar
    // codegen for `Vec<Option<T>>+as_str` is exercised across distinct
    // null/value layouts.
    let batch = vec![
        item.clone(),
        DeepWrappers {
            deep: vec![],
            vec_opt: vec![],
            triple: vec![],
        },
        DeepWrappers {
            deep: vec![vec![Status::Inactive], vec![]],
            vec_opt: vec![None, Some(Status::Active)],
            triple: vec![vec![vec![Status::Inactive]]],
        },
    ];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (3, 3));

    // Row 0: original `item` values.
    let deep_col = df_batch.column("deep").unwrap();
    if let AnyValue::List(outer) = deep_col.get(0).unwrap() {
        let lists: Vec<_> = outer.list().unwrap().into_iter().collect();
        assert_eq!(lists.len(), 2);
        let row0_a: Vec<Option<&str>> = lists[0]
            .as_ref()
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(row0_a, vec![Some("ACTIVE"), Some("INACTIVE")]);
    } else {
        panic!("expected list for batch deep[0]");
    }
    // Row 1: empty outer list.
    if let AnyValue::List(outer) = deep_col.get(1).unwrap() {
        assert_eq!(outer.list().unwrap().into_iter().count(), 0);
    } else {
        panic!("expected list for batch deep[1]");
    }
    // Row 2: distinct content from row 0.
    if let AnyValue::List(outer) = deep_col.get(2).unwrap() {
        let lists: Vec<_> = outer.list().unwrap().into_iter().collect();
        assert_eq!(lists.len(), 2);
        let row2_a: Vec<Option<&str>> = lists[0]
            .as_ref()
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(row2_a, vec![Some("INACTIVE")]);
        let row2_b: Vec<Option<&str>> = lists[1]
            .as_ref()
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(row2_b, vec![] as Vec<Option<&str>>);
    } else {
        panic!("expected list for batch deep[2]");
    }

    let vec_opt_col = df_batch.column("vec_opt").unwrap();
    if let AnyValue::List(inner) = vec_opt_col.get(0).unwrap() {
        let vals: Vec<Option<&str>> = inner.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![Some("ACTIVE"), None, Some("INACTIVE")]);
    } else {
        panic!("expected list for batch vec_opt[0]");
    }
    if let AnyValue::List(inner) = vec_opt_col.get(1).unwrap() {
        assert_eq!(inner.str().unwrap().into_iter().count(), 0);
    } else {
        panic!("expected list for batch vec_opt[1]");
    }
    if let AnyValue::List(inner) = vec_opt_col.get(2).unwrap() {
        let vals: Vec<Option<&str>> = inner.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![None, Some("ACTIVE")]);
    } else {
        panic!("expected list for batch vec_opt[2]");
    }
}
