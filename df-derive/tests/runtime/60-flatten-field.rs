#![allow(dead_code)]

use std::sync::Arc;

use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;

#[derive(ToDataFrame, Clone)]
struct Price {
    amount: i64,
    currency: String,
}

#[derive(ToDataFrame, Clone)]
struct OptionContractKey {
    underlying: String,
    side: String,
    strike: Price,
    expiration_date: String,
}

#[derive(ToDataFrame, Clone)]
struct ContractRow {
    #[df_derive(flatten)]
    key: OptionContractKey,
    contract_instrument: Option<String>,
    price: Option<Price>,
    contracts: Vec<OptionContractKey>,
}

#[derive(ToDataFrame, Clone)]
struct PrefixedRow {
    #[df_derive(flatten(prefix = "contract"))]
    key: OptionContractKey,
    side: String,
}

#[derive(ToDataFrame, Clone)]
struct Small {
    value: i32,
    label: String,
}

#[derive(ToDataFrame, Clone)]
struct PointerRow {
    #[df_derive(flatten)]
    boxed: Box<Small>,
    #[df_derive(flatten(prefix = "arc"))]
    arc: Arc<Small>,
}

#[derive(ToDataFrame, Clone)]
struct GenericFlatten<T> {
    id: u32,
    #[df_derive(flatten)]
    payload: T,
}

#[derive(ToDataFrame, Clone)]
struct CollisionChild {
    id: i32,
}

#[derive(ToDataFrame, Clone)]
struct ParentCollision {
    id: i32,
    #[df_derive(flatten)]
    child: CollisionChild,
}

#[derive(ToDataFrame, Clone)]
struct TwoFlattenCollision {
    #[df_derive(flatten)]
    left: CollisionChild,
    #[df_derive(flatten)]
    right: CollisionChild,
}

#[derive(ToDataFrame, Clone)]
struct PrefixCollision {
    contract: CollisionChild,
    #[df_derive(flatten(prefix = "contract"))]
    key: CollisionChild,
}

fn column_names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .into_iter()
        .map(|name| name.as_str().to_owned())
        .collect()
}

fn schema_names<T: ToDataFrame>() -> Vec<String> {
    T::schema()
        .unwrap()
        .iter()
        .map(|(name, _)| name.as_str().to_owned())
        .collect()
}

fn key(
    underlying: &str,
    side: &str,
    amount: i64,
    currency: &str,
    expiration_date: &str,
) -> OptionContractKey {
    OptionContractKey {
        underlying: underlying.to_owned(),
        side: side.to_owned(),
        strike: Price {
            amount,
            currency: currency.to_owned(),
        },
        expiration_date: expiration_date.to_owned(),
    }
}

fn assert_compute_error_contains<T>(result: PolarsResult<T>, expected: &str) {
    let Err(err) = result else {
        panic!("expected ComputeError containing `{expected}`");
    };
    match err {
        PolarsError::ComputeError(msg) => assert!(
            msg.contains(expected),
            "unexpected ComputeError message: {msg}"
        ),
        other => panic!("expected ComputeError containing `{expected}`, got {other:?}"),
    }
}

#[test]
fn runtime_semantics() {
    let rows = vec![ContractRow {
        key: key("AAPL", "call", 200, "USD", "2026-06-19"),
        contract_instrument: Some("AAPL260619C00200000".to_owned()),
        price: Some(Price {
            amount: 12,
            currency: "USD".to_owned(),
        }),
        contracts: vec![key("MSFT", "put", 300, "USD", "2026-09-18")],
    }];

    let expected = vec![
        "underlying",
        "side",
        "strike.amount",
        "strike.currency",
        "expiration_date",
        "contract_instrument",
        "price.amount",
        "price.currency",
        "contracts.underlying",
        "contracts.side",
        "contracts.strike.amount",
        "contracts.strike.currency",
        "contracts.expiration_date",
    ];
    assert_eq!(schema_names::<ContractRow>(), expected);

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, expected.len()));
    assert_eq!(column_names(&df), expected);
    assert_eq!(
        df.column("underlying").unwrap().get(0).unwrap(),
        AnyValue::String("AAPL"),
    );
    assert_eq!(
        df.column("strike.amount").unwrap().get(0).unwrap(),
        AnyValue::Int64(200),
    );
    assert!(df.column("key.underlying").is_err());
    assert!(df.column("contracts.underlying").is_ok());

    let empty = ContractRow::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, expected.len()));
    assert_eq!(column_names(&empty), expected);

    let prefixed = PrefixedRow {
        key: key("SPY", "put", 400, "USD", "2026-12-18"),
        side: "parent-side".to_owned(),
    };
    let prefixed_df = prefixed.to_dataframe().unwrap();
    assert_eq!(
        column_names(&prefixed_df),
        vec![
            "contract.underlying",
            "contract.side",
            "contract.strike.amount",
            "contract.strike.currency",
            "contract.expiration_date",
            "side",
        ],
    );

    let pointer = PointerRow {
        boxed: Box::new(Small {
            value: 1,
            label: "boxed".to_owned(),
        }),
        arc: Arc::new(Small {
            value: 2,
            label: "arc".to_owned(),
        }),
    };
    let pointer_df = pointer.to_dataframe().unwrap();
    assert_eq!(
        column_names(&pointer_df),
        vec!["value", "label", "arc.value", "arc.label"],
    );

    let generic = GenericFlatten {
        id: 9,
        payload: Small {
            value: 10,
            label: "payload".to_owned(),
        },
    };
    let generic_df = generic.to_dataframe().unwrap();
    assert_eq!(column_names(&generic_df), vec!["id", "value", "label"]);

    assert_compute_error_contains(ParentCollision::schema(), "duplicate column `id`");
    assert_compute_error_contains(
        ParentCollision {
            id: 1,
            child: CollisionChild { id: 2 },
        }
        .to_dataframe(),
        "duplicate column `id`",
    );
    assert_compute_error_contains(TwoFlattenCollision::schema(), "duplicate column `id`");
    assert_compute_error_contains(PrefixCollision::schema(), "duplicate column `contract.id`");
}
