use df_derive::ToDataFrame;
use polars::prelude::{DataFrame, DataType, PolarsResult};
use rust_decimal::Decimal;

mod row_traits {
    use super::*;

    pub trait MyToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
    }
}

mod batch_traits {
    use super::*;

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let refs: Vec<&Self> = items.iter().collect();
            Self::columnar_from_refs(&refs)
        }

        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
    }

    pub trait MyToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> MyToDataFrameVec for [T]
    where
        T: Columnar + super::row_traits::MyToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as super::row_traits::MyToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }
}

mod decimal_traits {
    pub trait MyDecimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }

    impl MyDecimal128Encode for rust_decimal::Decimal {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
            if self.scale() == target_scale {
                Some(self.mantissa())
            } else {
                None
            }
        }
    }
}

#[derive(ToDataFrame)]
#[df_derive(
    columnar = "batch_traits::Columnar",
    decimal128_encode = "decimal_traits::MyDecimal128Encode",
    trait = "row_traits::MyToDataFrame",
)]
struct ReversedOverrides {
    #[df_derive(decimal(precision = 18, scale = 2))]
    amount: Decimal,
}

fn main() {
    use batch_traits::MyToDataFrameVec;

    let row = ReversedOverrides {
        amount: Decimal::new(1234, 2),
    };
    let df_single = row_traits::MyToDataFrame::to_dataframe(&row).unwrap();
    assert_eq!(df_single.shape(), (1, 1));
    assert_eq!(df_single.get_column_names(), &["amount"]);

    let rows = vec![
        ReversedOverrides {
            amount: Decimal::new(1234, 2),
        },
        ReversedOverrides {
            amount: Decimal::new(5678, 2),
        },
    ];
    let df_batch = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.shape(), (2, 1));
    assert_eq!(df_batch.get_column_names(), &["amount"]);
}
