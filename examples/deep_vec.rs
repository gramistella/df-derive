//! Demonstrates deep `Vec<Vec<Vec<T>>>` nesting.
//!
//! Each `Vec` layer becomes a `LargeListArray` wrap, so a triple-Vec column
//! materializes as `List<List<List<T>>>` in the output frame. The derive
//! fuses the wrapper layers into a single bulk emission, so per-row work is
//! O(total leaf count) rather than O(layer count * leaf count).

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

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
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

    let schema = <Tensor as crate::dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (each Vec layer becomes a List wrap):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
