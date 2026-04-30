//! Generic structs (v0.3.0).
//!
//! `#[derive(ToDataFrame)]` accepts type parameters, default type parameters,
//! and multiple generics. The macro injects `ToDataFrame + Columnar + Clone`
//! bounds on every type parameter, so any concrete instantiation must satisfy
//! those traits. The unit type `()` can be used as a payload to contribute
//! zero columns.

use df_derive::ToDataFrame;

#[allow(dead_code)]
mod dataframe {
    use polars::prelude::{AnyValue, DataFrame, DataType, NamedFrom, PolarsResult, Series};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
        fn to_inner_values(&self) -> PolarsResult<Vec<AnyValue<'static>>> {
            let df = self.to_dataframe()?;
            let row = df.get(0).unwrap_or_default();
            Ok(row.into_iter().map(AnyValue::into_static).collect())
        }
    }

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let refs: Vec<&Self> = items.iter().collect();
            Self::columnar_from_refs(&refs)
        }
        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
    }

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar + ToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as ToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }

    // The unit type `()` contributes zero columns. The DataFrame must still
    // carry the right number of rows, so we attach a temporary dummy column
    // and drop it before returning.
    impl ToDataFrame for () {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            let dummy = Series::new("_dummy".into(), &[0i32]);
            let mut df = DataFrame::new_infer_height(vec![dummy.into()])?;
            df.drop_in_place("_dummy")?;
            Ok(df)
        }
        fn empty_dataframe() -> PolarsResult<DataFrame> {
            DataFrame::new_infer_height(vec![])
        }
        fn schema() -> PolarsResult<Vec<(String, DataType)>> {
            Ok(Vec::new())
        }
    }

    impl Columnar for () {
        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
            let dummy = Series::new_empty("_dummy".into(), &DataType::Null)
                .extend_constant(AnyValue::Null, items.len())?;
            let mut df = DataFrame::new_infer_height(vec![dummy.into()])?;
            df.drop_in_place("_dummy")?;
            Ok(df)
        }
    }
}

use crate::dataframe::{ToDataFrame, ToDataFrameVec};

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Meta {
    timestamp: i64,
    note: String,
}

// No explicit `T: Clone` bounds below — `#[derive(Clone)]` adds its own, and
// the derive macro auto-injects `ToDataFrame + Columnar + Clone` on every type
// parameter for the impl blocks it generates.

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Wrapper<T> {
    id: u32,
    payload: T,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct DefaultMeta<M = ()> {
    value: i32,
    meta: M,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Pair<A, B> {
    name: String,
    left: A,
    right: B,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct OptWrapper<T> {
    id: u32,
    payload: Option<T>,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
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

    // 3. Multiple generic parameters. Each parameter gets the standard bound
    //    set (`ToDataFrame + Columnar + Clone`) injected by the macro.
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

    // 4. Option<T> and Vec<T> for generic T. Both go through the bulk
    //    columnar path (gather/scatter for Option, flatten + slice for Vec)
    //    when emitted by the derive.
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
