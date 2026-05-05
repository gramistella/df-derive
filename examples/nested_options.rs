//! Demonstrates `Option<Option<Struct>>` field handling.
//!
//! Polars carries one validity bit per nullable level, so `Some(None)` and
//! `None` collapse to the same `Null` in the output frame. This example
//! shows how the derive flattens the nested-struct columns and how both
//! outer-`None` and inner-`None` rows surface as nulls.
//!
//! Uses `df-derive-runtime` for the canonical trait module — the macro accepts
//! any user-defined module at this path; see `quickstart.rs` for the inline form.

use df_derive::ToDataFrame;
use df_derive_runtime::dataframe;
use df_derive_runtime::dataframe::ToDataFrameVec;

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
struct Profile {
    handle: String,
    score: f64,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
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

    let schema = <User as dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (profile fields are flattened with dot notation):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
