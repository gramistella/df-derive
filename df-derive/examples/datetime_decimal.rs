// `rust_decimal::Decimal` and `chrono::DateTime<Tz>` fields need no attribute.
//
// A decimal is stored as `Decimal(38, 10)`. A `DateTime<Tz>` is an instant,
// so its column is `Datetime(Milliseconds, UTC)`; a `NaiveDateTime` would
// give the same column with no time zone.

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
    println!("\nSchema (amount = Decimal(38, 10), ts = Datetime(Milliseconds, UTC)):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
