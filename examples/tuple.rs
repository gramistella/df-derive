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
struct SimpleTuple(i32, String, f64);

#[allow(clippy::approx_constant)]
fn main() -> polars::prelude::PolarsResult<()> {
    let tuple = SimpleTuple(42, "Hello".to_string(), 3.14);

    let df = <SimpleTuple as crate::dataframe::ToDataFrame>::to_dataframe(&tuple)?;
    println!("Tuple struct DataFrame:");
    println!("{df}");

    // Show schema to demonstrate column naming
    let schema = <SimpleTuple as crate::dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (columns: field_0 (Int32), field_1 (String), field_2 (Float64)):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
