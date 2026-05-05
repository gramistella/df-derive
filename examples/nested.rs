// Uses `df-derive-runtime` for the canonical trait module — the macro accepts
// any user-defined module at this path; see `quickstart.rs` for the inline form.

use df_derive::ToDataFrame;
use df_derive_runtime::dataframe;

#[derive(ToDataFrame)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
struct Address {
    street: String,
    city: String,
    zip: u32,
}

#[derive(ToDataFrame)]
#[df_derive(trait = "df_derive_runtime::dataframe::ToDataFrame")]
struct Person {
    name: String,
    age: u32,
    address: Address,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let person = Person {
        name: "John Doe".to_string(),
        age: 30,
        address: Address {
            street: "123 Main St".to_string(),
            city: "New York".to_string(),
            zip: 10001,
        },
    };

    let df = <Person as dataframe::ToDataFrame>::to_dataframe(&person)?;
    println!("Nested struct DataFrame:");
    println!("{df}");

    // Show schema to demonstrate column naming
    let schema = <Person as dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (columns: name, age, address.street, address.city, address.zip):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
