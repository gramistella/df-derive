//! Quick-start example using the default `df-derive` facade runtime.

use df_derive::prelude::*;

#[derive(ToDataFrame)]
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
    let df_single = t.to_dataframe()?;
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
