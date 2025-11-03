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
struct Address {
    street: String,
    city: String,
    zip: u32,
}

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
struct Person {
    name: String,
    age: u32,
    address: Address,
}

// Import the trait to make it available for nested structs
use crate::dataframe::ToDataFrame;

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

    let df = <Person as crate::dataframe::ToDataFrame>::to_dataframe(&person)?;
    println!("Nested struct DataFrame:");
    println!("{df}");

    // Show schema to demonstrate column naming
    let schema = <Person as crate::dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (columns: name, age, address.street, address.city, address.zip):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
