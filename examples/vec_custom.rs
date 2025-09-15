use df_derive::ToDataFrame;

#[allow(dead_code)]
mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
    }

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
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
struct Quote {
    ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct MarketData {
    symbol: String,
    quotes: Vec<Quote>,
}

// Import the trait to make it available for nested structs
use crate::dataframe::ToDataFrame;

fn main() -> polars::prelude::PolarsResult<()> {
    let market_data = MarketData {
        symbol: "AAPL".to_string(),
        quotes: vec![
            Quote {
                ts: 1640995200,
                open: 182.01,
                high: 182.54,
                low: 179.55,
                close: 180.33,
                volume: 1000000,
            },
            Quote {
                ts: 1641081600,
                open: 180.33,
                high: 181.12,
                low: 179.20,
                close: 180.95,
                volume: 1200000,
            },
            Quote {
                ts: 1641168000,
                open: 180.95,
                high: 182.20,
                low: 180.10,
                close: 181.50,
                volume: 900000,
            },
        ],
    };

    let df = <MarketData as crate::dataframe::ToDataFrame>::to_dataframe(&market_data)?;
    println!("Vec of custom structs DataFrame:");
    println!("{}", df);

    // Show schema to demonstrate column naming
    let schema = <MarketData as crate::dataframe::ToDataFrame>::schema()?;
    println!(
        "\nSchema (columns include: symbol, quotes.ts, quotes.open, quotes.high, ... (each a List)):"
    );
    for (name, dtype) in schema {
        println!("  {}: {:?}", name, dtype);
    }

    Ok(())
}
