//! Generic structs.
//!
//! `#[derive(ToDataFrame)]` accepts type parameters, default type parameters,
//! and multiple generics. The generated impls require each field's type to be
//! `Columnar`, so a parameter can stand for any supported type: a derived
//! struct, a number, a string, a wrapper around one. The unit type `()` is a
//! payload that contributes no columns.

use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame as _, ToDataFrameVec as _};

#[derive(ToDataFrame, Clone)]
struct Meta {
    timestamp: i64,
    note: String,
}

// No bounds are written below. `#[derive(Clone)]` adds its own `T: Clone`,
// and `ToDataFrame` bounds the impls it generates by the fields' types.

#[derive(ToDataFrame, Clone)]
struct Wrapper<T> {
    id: u32,
    payload: T,
}

#[derive(ToDataFrame, Clone)]
struct DefaultMeta<M = ()> {
    value: i32,
    meta: M,
}

#[derive(ToDataFrame, Clone)]
struct Pair<A, B> {
    name: String,
    left: A,
    right: B,
}

#[derive(ToDataFrame, Clone)]
struct OptWrapper<T> {
    id: u32,
    payload: Option<T>,
}

#[derive(ToDataFrame, Clone)]
struct VecWrapper<T> {
    id: u32,
    payload: Vec<T>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    // 1. Generic over a nested struct. Schema flattens to `id`, plus
    //    `payload.timestamp` and `payload.note` from `Meta`.
    let row = Wrapper {
        id: 1,
        payload: Meta {
            timestamp: 1_700_000_000,
            note: "hello".into(),
        },
    };
    println!("Wrapper<Meta> (single):\n{}\n", row.to_dataframe()?);

    let rows = vec![
        Wrapper {
            id: 1,
            payload: Meta {
                timestamp: 1_700_000_000,
                note: "first".into(),
            },
        },
        Wrapper {
            id: 2,
            payload: Meta {
                timestamp: 1_700_000_100,
                note: "second".into(),
            },
        },
    ];
    println!(
        "Wrapper<Meta> (batch):\n{}\n",
        rows.as_slice().to_dataframe()?
    );

    // 2. Unit payload. `Wrapper<()>` and `DefaultMeta` (default M = ())
    //    contribute no extra columns — useful for "tagged" rows that carry no
    //    extra payload but share a schema with richer instantiations.
    let unit = Wrapper { id: 7, payload: () };
    println!("Wrapper<()> (unit payload):\n{}\n", unit.to_dataframe()?);

    let default_meta: DefaultMeta = DefaultMeta {
        value: 42,
        meta: (),
    };
    println!(
        "DefaultMeta (M defaults to ()):\n{}\n",
        default_meta.to_dataframe()?
    );

    // 3. Several generic parameters. A field of type `A` needs
    //    `A: Columnar`, and the same goes for `B`.
    let pair = Pair {
        name: "trade-1".into(),
        left: Meta {
            timestamp: 1,
            note: "buy".into(),
        },
        right: Meta {
            timestamp: 2,
            note: "sell".into(),
        },
    };
    println!("Pair<Meta, Meta>:\n{}\n", pair.to_dataframe()?);

    // 4. Option<T> and Vec<T> for generic T: a nullable payload and a list
    //    payload, whatever `T` is.
    let opts = vec![
        OptWrapper {
            id: 1,
            payload: Some(Meta {
                timestamp: 10,
                note: "present".into(),
            }),
        },
        OptWrapper {
            id: 2,
            payload: None,
        },
    ];
    println!(
        "OptWrapper<Meta> (batch):\n{}\n",
        opts.as_slice().to_dataframe()?
    );

    let vecs = vec![
        VecWrapper {
            id: 1,
            payload: vec![
                Meta {
                    timestamp: 1,
                    note: "a".into(),
                },
                Meta {
                    timestamp: 2,
                    note: "b".into(),
                },
            ],
        },
        VecWrapper {
            id: 2,
            payload: vec![],
        },
    ];
    println!(
        "VecWrapper<Meta> (batch):\n{}\n",
        vecs.as_slice().to_dataframe()?
    );

    Ok(())
}
