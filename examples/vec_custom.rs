// Uses `df-derive-runtime` for the canonical trait module — the macro accepts
// any user-defined module at this path; see `quickstart.rs` for the inline form.

use df_derive::ToDataFrame;
use df_derive_runtime::dataframe;

#[derive(ToDataFrame)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
struct Quote {
    ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

#[derive(ToDataFrame)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
struct MarketData {
    symbol: String,
    quotes: Vec<Quote>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let market_data = MarketData {
        symbol: "AAPL".to_string(),
        quotes: vec![
            Quote {
                ts: 1_640_995_200,
                open: 182.01,
                high: 182.54,
                low: 179.55,
                close: 180.33,
                volume: 1_000_000,
            },
            Quote {
                ts: 1_641_081_600,
                open: 180.33,
                high: 181.12,
                low: 179.20,
                close: 180.95,
                volume: 1_200_000,
            },
            Quote {
                ts: 1_641_168_000,
                open: 180.95,
                high: 182.20,
                low: 180.10,
                close: 181.50,
                volume: 900_000,
            },
        ],
    };

    let df = <MarketData as dataframe::ToDataFrame>::to_dataframe(&market_data)?;
    println!("Vec of custom structs DataFrame:");
    println!("{df}");

    // Show schema to demonstrate column naming
    let schema = <MarketData as dataframe::ToDataFrame>::schema()?;
    println!(
        "\nSchema (columns include: symbol, quotes.ts, quotes.open, quotes.high, ... (each a List)):"
    );
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
