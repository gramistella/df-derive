//! Quick-start example. This file is intentionally self-contained: it
//! defines the `dataframe` trait module inline so it works without any
//! runtime crate dependency. The macro accepts any user-defined module at
//! the path you point `#[df_derive(trait = "...")]` at — that's the whole
//! reason `df-derive` is decoupled from a fixed runtime.
//!
//! For the "skip the boilerplate" path, depend on `df-derive-runtime` and
//! point the derive at `df_derive_runtime::dataframe::ToDataFrame` instead.
//! The other examples in this directory show that shape.

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
#[df_derive(trait = "crate::dataframe::ToDataFrame")] // Columnar path auto-infers to crate::dataframe::Columnar
struct Trade {
    symbol: String,
    price: f64,
    size: u64,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let t = Trade {
        symbol: "AAPL".into(),
        price: 187.23,
        size: 100,
    };
    let df_single = <Trade as crate::dataframe::ToDataFrame>::to_dataframe(&t)?;
    println!("Single trade DataFrame:");
    println!("{df_single}");

    let trades = vec![
        Trade {
            symbol: "AAPL".into(),
            price: 187.23,
            size: 100,
        },
        Trade {
            symbol: "MSFT".into(),
            price: 411.61,
            size: 200,
        },
    ];
    let df_batch = trades.as_slice().to_dataframe()?;
    println!("\nBatch trades DataFrame:");
    println!("{df_batch}");

    Ok(())
}
