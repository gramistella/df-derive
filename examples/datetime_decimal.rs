use chrono::{DateTime, Utc};
use df_derive::ToDataFrame;
use rust_decimal::Decimal;

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
struct TxRecord {
    amount: Decimal,
    ts: DateTime<Utc>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let tx = TxRecord {
        amount: Decimal::new(12345, 2), // 123.45
        ts: Utc::now(),
    };

    let df = <TxRecord as crate::dataframe::ToDataFrame>::to_dataframe(&tx)?;
    println!("DateTime and Decimal DataFrame:");
    println!("{df}");

    // Show schema to demonstrate data types
    let schema = <TxRecord as crate::dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (amount = Decimal(38, 10), ts = Datetime(Milliseconds, None)):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
