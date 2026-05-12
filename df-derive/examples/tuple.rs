// Uses the default `df-derive` facade runtime.

use df_derive::ToDataFrame;
use df_derive::dataframe;

#[derive(ToDataFrame)]
struct SimpleTuple(i32, String, f64);

#[allow(clippy::approx_constant)]
fn main() -> polars::prelude::PolarsResult<()> {
    let tuple = SimpleTuple(42, "Hello".to_string(), 3.14);

    let df = <SimpleTuple as dataframe::ToDataFrame>::to_dataframe(&tuple)?;
    println!("Tuple struct DataFrame:");
    println!("{df}");

    // Show schema to demonstrate column naming
    let schema = <SimpleTuple as dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (columns: field_0 (Int32), field_1 (String), field_2 (Float64)):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
