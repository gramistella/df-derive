//! Demonstrates multi-`Option` wrappers above a `Vec`.
//!
//! Consecutive `Option` layers above a `Vec` collapse to a single list-level
//! validity bit (Polars semantics) — `Some(None)` and the outer `None` are
//! indistinguishable in the resulting column. This is a documented contract,
//! not a bug.
//!
//! The example walks two shapes:
//!   1. `Option<Option<Vec<i32>>>` — a primitive leaf, where the column dtype
//!      is `List<Int32>` and three different source values produce two
//!      distinct rows: a populated list, and a null cell shared by both
//!      `Some(None)` and `None`.
//!   2. `Option<Option<Vec<Option<i64>>>>` — a leaf with its own per-element
//!      `Option`. The outer `Option<Option<...>>` still collapses to one
//!      list-level validity bit, while the *inner* `Option` survives as
//!      per-element nullability inside the list.
//!
//! `Vec<T>` itself is not nullable — only the outer-list validity bit can
//! make a row null. So `Some(Some(vec![]))` is observably distinct from
//! `None` / `Some(None)`: the former is an empty list, the latter is null.
//!
//! Uses the default `df-derive` facade runtime.

use df_derive::ToDataFrame;
use df_derive::dataframe;
use df_derive::dataframe::ToDataFrameVec;

#[derive(ToDataFrame, Clone)]
struct Sample {
    id: u32,
    // The example's whole point is to demonstrate this shape.
    #[allow(clippy::option_option)]
    counts: Option<Option<Vec<i32>>>,
    // Inner `Option` survives as per-element validity; the outer
    // `Option<Option<...>>` still collapses to one list-level validity bit.
    #[allow(clippy::option_option)]
    readings: Option<Option<Vec<Option<i64>>>>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let samples = vec![
        // Some(Some(populated)) — list with values, inner Some/None survives.
        Sample {
            id: 1,
            counts: Some(Some(vec![10, 20, 30])),
            readings: Some(Some(vec![Some(100), None, Some(300)])),
        },
        // Some(Some(empty)) — empty list, NOT null. Distinguishable from
        // the rows below.
        Sample {
            id: 2,
            counts: Some(Some(vec![])),
            readings: Some(Some(vec![])),
        },
        // Some(None) — collapses to null (outer-list validity bit clear).
        Sample {
            id: 3,
            counts: Some(None),
            readings: Some(None),
        },
        // None — also null. Indistinguishable from row 3 in the column.
        Sample {
            id: 4,
            counts: None,
            readings: None,
        },
    ];

    let df = samples.as_slice().to_dataframe()?;
    println!("Multi-Option-over-Vec DataFrame:");
    println!("{df}");

    let schema = <Sample as dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (consecutive Options collapse to one list-level validity bit):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    println!(
        "\nRows 3 (Some(None)) and 4 (None) both surface as the same null cell — \
         Polars carries one validity bit per nullable level, so the two source \
         values are indistinguishable in the column.",
    );

    Ok(())
}
