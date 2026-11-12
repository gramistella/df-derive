//! Demonstrates deep `Vec<Vec<Vec<T>>>` nesting.
//!
//! Each `Vec` layer adds one `List` around the column's type, so a triple-Vec
//! field is a `List(List(List(T)))` column. The elements are written once,
//! into one buffer; each layer only adds its list offsets.

use df_derive::ToDataFrame;
use df_derive::dataframe;
use df_derive::dataframe::ToDataFrameVec;

#[derive(ToDataFrame)]
struct Tensor {
    label: String,
    // List<List<List<f64>>> after derive: outer batches, sequences, frames.
    values: Vec<Vec<Vec<f64>>>,
    // List<List<i32>> for a smaller comparison column.
    counts: Vec<Vec<i32>>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let tensors = vec![
        Tensor {
            label: "alpha".into(),
            values: vec![vec![vec![1.0, 2.0], vec![3.0]], vec![vec![4.0, 5.0, 6.0]]],
            counts: vec![vec![1, 2, 3], vec![]],
        },
        Tensor {
            label: "beta".into(),
            values: vec![],
            counts: vec![vec![10]],
        },
    ];

    let df = tensors.as_slice().to_dataframe()?;
    println!("Deep Vec<Vec<Vec<T>>> DataFrame:");
    println!("{df}");

    let schema = <Tensor as dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (each Vec layer becomes a List wrap):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
