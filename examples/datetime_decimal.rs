use chrono::{DateTime, Utc};
use df_derive::ToDataFrame;
use rust_decimal::Decimal;

#[allow(dead_code)]
mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
    }

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let refs: Vec<&Self> = items.iter().collect();
            Self::columnar_from_refs(&refs)
        }
        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
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

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }

    impl Decimal128Encode for rust_decimal::Decimal {
        #[inline]
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
            let source_scale = self.scale();
            let mantissa: i128 = self.mantissa();
            if source_scale == target_scale {
                return Some(mantissa);
            }
            if source_scale < target_scale {
                let diff = target_scale - source_scale;
                let pow = 10i128.pow(diff);
                return mantissa.checked_mul(pow);
            }
            let diff = source_scale - target_scale;
            let pow = 10i128.pow(diff).cast_unsigned();
            let neg = mantissa < 0;
            let abs = mantissa.unsigned_abs();
            let q = (abs / pow).cast_signed();
            let r = abs % pow;
            let half = pow / 2;
            let rounded = match r.cmp(&half) {
                std::cmp::Ordering::Greater => q + 1,
                std::cmp::Ordering::Less => q,
                std::cmp::Ordering::Equal => q + (q & 1),
            };
            Some(if neg { -rounded } else { rounded })
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
