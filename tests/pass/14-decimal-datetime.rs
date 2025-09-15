use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

use chrono::{DateTime, TimeZone, Utc};
use polars::prelude::*;
use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct TxRecord {
    amount: Decimal,
    ts: DateTime<Utc>,
}

fn main() {
    println!("────────────────────────────────────────────────────────");
    println!("🔎 Testing Decimal and DateTime<Utc> handling...");

    let ts = Utc.timestamp_millis_opt(1_700_000_000_123).single().unwrap();
    let tx = TxRecord {
        amount: Decimal::new(12345, 2), // 123.45
        ts,
    };

    // Schema should advertise high-fidelity dtypes
    let schema = TxRecord::schema().unwrap();
    let amount_dtype = schema
        .iter()
        .find(|(n, _)| *n == "amount")
        .map(|(_, dt)| dt.clone())
        .expect("amount dtype missing");
    let ts_dtype = schema
        .iter()
        .find(|(n, _)| *n == "ts")
        .map(|(_, dt)| dt.clone())
        .expect("ts dtype missing");

    println!("📋 Schema dtypes:");
    println!("  - amount: {:?}", amount_dtype);
    println!("  - ts: {:?}", ts_dtype);
    assert_eq!(amount_dtype, DataType::Decimal(Some(38), Some(10)));
    assert_eq!(ts_dtype, DataType::Datetime(TimeUnit::Milliseconds, None));

    // Convert to DataFrame and validate columns
    let df = tx.to_dataframe().unwrap();
    assert_eq!(df.height(), 1);
    assert!(df.get_column_names().iter().any(|c| c.as_str() == "amount"));
    assert!(df.get_column_names().iter().any(|c| c.as_str() == "ts"));

    println!("\n📊 Resulting DataFrame:");
    println!("{}", df);

    // DateTime<Utc> should materialize as Datetime(Milliseconds, None)
    let ts_runtime_dtype = df.column("ts").unwrap().dtype().clone();
    println!("\n🧪 Runtime dtype checks:");
    println!("  - ts runtime dtype: {:?}", ts_runtime_dtype);
    assert_eq!(ts_runtime_dtype, DataType::Datetime(TimeUnit::Milliseconds, None));

    // Decimal should materialize as Decimal; verify dtype
    let amount_runtime_dtype = df.column("amount").unwrap().dtype().clone();
    println!("  - amount runtime dtype: {:?}", amount_runtime_dtype);
    assert_eq!(amount_runtime_dtype, DataType::Decimal(Some(38), Some(10)));

    println!("\n✅ Decimal and DateTime<Utc> test completed successfully!");
}


