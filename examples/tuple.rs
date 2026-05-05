// Uses `df-derive-runtime` for the canonical trait module — the macro accepts
// any user-defined module at this path; see `quickstart.rs` for the inline form.

use df_derive::ToDataFrame;
use df_derive_runtime::dataframe;

#[derive(ToDataFrame)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
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
