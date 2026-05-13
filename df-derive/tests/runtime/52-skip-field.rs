#![allow(dead_code)]

use std::collections::HashMap;

use df_derive::ToDataFrame;
use polars::prelude::*;

use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

struct CacheOnly {
    value: i32,
}

#[derive(ToDataFrame)]
struct WithSkippedField {
    id: i32,
    #[df_derive(skip)]
    cache: HashMap<String, CacheOnly>,
    name: String,
}

#[derive(ToDataFrame)]
struct AllSkipped {
    #[df_derive(skip)]
    cache: HashMap<String, CacheOnly>,
    #[df_derive(skip)]
    payload: CacheOnly,
}

#[derive(ToDataFrame)]
struct SkippedGeneric<T> {
    id: i32,
    #[df_derive(skip)]
    payload: T,
}

#[derive(ToDataFrame)]
struct TupleSkip(i32, #[df_derive(skip)] String, bool);

fn column_names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .into_iter()
        .map(|name| name.as_str().to_owned())
        .collect()
}

#[test]
fn runtime_semantics() {
    let rows = vec![
        WithSkippedField {
            id: 1,
            cache: HashMap::from([("a".to_owned(), CacheOnly { value: 10 })]),
            name: "alpha".to_owned(),
        },
        WithSkippedField {
            id: 2,
            cache: HashMap::new(),
            name: "beta".to_owned(),
        },
    ];
    let schema = WithSkippedField::schema().unwrap();
    assert_eq!(
        schema
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["id", "name"],
    );

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (2, 2));
    assert_eq!(column_names(&df), vec!["id".to_owned(), "name".to_owned()]);
    assert!(df.column("cache").is_err());
    assert_eq!(df.column("id").unwrap().get(0).unwrap(), AnyValue::Int32(1));
    assert_eq!(
        df.column("name").unwrap().get(1).unwrap(),
        AnyValue::String("beta"),
    );

    let empty = WithSkippedField::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 2));
    assert_eq!(
        column_names(&empty),
        vec!["id".to_owned(), "name".to_owned()]
    );

    let all = AllSkipped {
        cache: HashMap::new(),
        payload: CacheOnly { value: 99 },
    };
    let all_df = all.to_dataframe().unwrap();
    assert_eq!(all_df.shape(), (1, 0));
    assert!(AllSkipped::schema().unwrap().is_empty());
    assert_eq!(AllSkipped::empty_dataframe().unwrap().shape(), (0, 0));
    assert_eq!(
        vec![
            AllSkipped {
                cache: HashMap::new(),
                payload: CacheOnly { value: 1 },
            },
            all,
        ]
        .as_slice()
        .to_dataframe()
        .unwrap()
        .shape(),
        (2, 0),
    );

    struct NoTraits;
    let generic_rows = [SkippedGeneric {
        id: 7,
        payload: NoTraits,
    }];
    let generic_df = generic_rows.as_slice().to_dataframe().unwrap();
    assert_eq!(generic_df.shape(), (1, 1));
    assert_eq!(
        generic_df.column("id").unwrap().get(0).unwrap(),
        AnyValue::Int32(7),
    );

    let tuple = TupleSkip(42, "ignored".to_owned(), true);
    let tuple_df = tuple.to_dataframe().unwrap();
    assert_eq!(tuple_df.shape(), (1, 2));
    assert_eq!(
        column_names(&tuple_df),
        vec!["field_0".to_owned(), "field_2".to_owned()]
    );
    assert!(tuple_df.column("field_1").is_err());
    assert_eq!(
        tuple_df.column("field_0").unwrap().get(0).unwrap(),
        AnyValue::Int32(42),
    );
    assert_eq!(
        tuple_df.column("field_2").unwrap().get(0).unwrap(),
        AnyValue::Boolean(true),
    );
}
