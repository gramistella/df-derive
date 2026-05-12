use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

// `as_str` on a raw `String` field is allowed and behaves identically to
// omitting the attribute. The codegen routes through the same deref-coerce
// branch the bare-`String` borrowing path uses, so output equality must hold
// for both the row-wise and columnar paths.

#[derive(ToDataFrame, Clone)]
struct WithAttr {
    #[df_derive(as_str)]
    s: String,
    #[df_derive(as_str)]
    o: Option<String>,
    #[df_derive(as_str)]
    v: Vec<String>,
    #[df_derive(as_str)]
    ov: Option<Vec<String>>,
}

#[derive(ToDataFrame, Clone)]
struct Without {
    s: String,
    o: Option<String>,
    v: Vec<String>,
    ov: Option<Vec<String>>,
}

fn main() {
    println!("--- Testing redundant #[df_derive(as_str)] on raw String fields ---");

    let with = WithAttr {
        s: "hi".into(),
        o: Some("there".into()),
        v: vec!["a".into(), "b".into()],
        ov: Some(vec!["x".into(), "y".into()]),
    };
    let without = Without {
        s: "hi".into(),
        o: Some("there".into()),
        v: vec!["a".into(), "b".into()],
        ov: Some(vec!["x".into(), "y".into()]),
    };

    let df_with = with.to_dataframe().unwrap();
    let df_without = without.to_dataframe().unwrap();

    println!("With attribute:\n{}", df_with);
    println!("Without attribute:\n{}", df_without);

    assert_eq!(df_with.schema(), df_without.schema());
    // Row-wise input has no `None` values so plain `equals` is enough — and
    // it's a stricter check than `equals_missing` (the latter treats null ==
    // null, which would mask any spurious-null bug between the two paths).
    assert!(
        df_with.equals(&df_without),
        "row-wise output diverged between with/without `as_str` on String"
    );

    println!("🔄 Slice-batch (columnar) comparison...");
    let with_batch = vec![
        with.clone(),
        WithAttr {
            s: "abc".into(),
            o: None,
            v: vec![],
            ov: None,
        },
        with.clone(),
    ];
    let without_batch = vec![
        without.clone(),
        Without {
            s: "abc".into(),
            o: None,
            v: vec![],
            ov: None,
        },
        without.clone(),
    ];
    let mut df_with_batch = with_batch.as_slice().to_dataframe().unwrap();
    let mut df_without_batch = without_batch.as_slice().to_dataframe().unwrap();

    println!("With (batch):\n{}", df_with_batch);
    println!("Without (batch):\n{}", df_without_batch);

    assert_eq!(df_with_batch.schema(), df_without_batch.schema());

    // `equals_missing` treats null == null. The columnar batch contains an
    // explicit `None` row that exercises both `Option<String>` and
    // `Option<Vec<String>>` shapes — `equals` (without missing-aware
    // semantics) would reject any null, even when both sides agree.
    df_with_batch.align_chunks_par();
    df_without_batch.align_chunks_par();
    assert!(
        df_with_batch.equals_missing(&df_without_batch),
        "columnar output diverged between with/without `as_str` on String"
    );

    println!("\n✅ Redundant as_str on String matches plain String output exactly.");
}
