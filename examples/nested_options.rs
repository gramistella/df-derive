//! Demonstrates `Option<Option<Struct>>` field handling.
//!
//! Polars carries one validity bit per nullable level, so `Some(None)` and
//! `None` collapse to the same `Null` in the output frame. This example
//! shows how the derive flattens the nested-struct columns and how both
//! outer-`None` and inner-`None` rows surface as nulls.

use crate::dataframe::ToDataFrameVec;
use df_derive::ToDataFrame;

#[allow(dead_code)]
mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
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
}

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Profile {
    handle: String,
    score: f64,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct User {
    id: u32,
    // The example's whole point is to demonstrate this shape.
    #[allow(clippy::option_option)]
    profile: Option<Option<Profile>>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let users = vec![
        User {
            id: 1,
            profile: Some(Some(Profile {
                handle: "alice".into(),
                score: 87.5,
            })),
        },
        User {
            id: 2,
            // `Some(None)` and `None` are indistinguishable in the column.
            profile: Some(None),
        },
        User {
            id: 3,
            profile: None,
        },
    ];

    let df = users.as_slice().to_dataframe()?;
    println!("Option<Option<Struct>> DataFrame:");
    println!("{df}");

    let schema = <User as crate::dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (profile fields are flattened with dot notation):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
