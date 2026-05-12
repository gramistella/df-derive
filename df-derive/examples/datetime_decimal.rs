// Uses the default `df-derive` facade runtime.
// `df-derive` ships the reference `Decimal128Encode for
// rust_decimal::Decimal` impl behind the `rust_decimal` feature, which is
// enabled by default; the codegen finds it via the auto-inferred
// `df_derive::dataframe::Decimal128Encode` path.

use chrono::{DateTime, Utc};
use df_derive::ToDataFrame;
use df_derive::dataframe;
use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct TxRecord {
    amount: Decimal,
    ts: DateTime<Utc>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let tx = TxRecord {
        amount: Decimal::new(12345, 2), // 123.45
        ts: Utc::now(),
    };

    let df = <TxRecord as dataframe::ToDataFrame>::to_dataframe(&tx)?;
    println!("DateTime and Decimal DataFrame:");
    println!("{df}");

    // Show schema to demonstrate data types
    let schema = <TxRecord as dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (amount = Decimal(38, 10), ts = Datetime(Milliseconds, None)):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
