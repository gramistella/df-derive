use df_derive::ToDataFrame;
use polars::prelude::{DataFrame, PolarsResult, SchemaRef};

mod custom_runtime {
    use super::*;

    pub trait MyColumnar: Sized {
        fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>;
    }

    pub trait MyToDataFrame: MyColumnar {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            Self::encode(std::slice::from_ref(self))
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            Self::encode(&[] as &[Self])
        }

        fn schema() -> PolarsResult<SchemaRef> {
            Ok(Self::empty_dataframe()?.schema().clone())
        }
    }

    impl<T: MyColumnar> MyToDataFrame for T {}

    #[derive(Clone)]
    pub struct CustomDecimal(pub i128);

    pub trait MyDecimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }

    impl MyDecimal128Encode for CustomDecimal {
        fn try_to_i128_mantissa(&self, _target_scale: u32) -> Option<i128> {
            Some(self.0)
        }
    }

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: MyColumnar,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            <T as MyColumnar>::encode(self)
        }
    }
}

mod columnar_only_runtime {
    pub use super::custom_runtime::{
        MyColumnar as Columnar, MyDecimal128Encode as Decimal128Encode,
        MyToDataFrame as ToDataFrame,
    };
}

#[derive(ToDataFrame)]
#[df_derive(trait = "df_derive::dataframe::ToDataFrame")]
struct BuiltinTraitOnly {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(
    trait = "df_derive::dataframe::ToDataFrame",
    columnar = "df_derive::dataframe::Columnar"
)]
struct BuiltinTraitAndColumnar {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(
    trait = "custom_runtime::MyToDataFrame",
    columnar = "custom_runtime::MyColumnar"
)]
struct CustomTraitAndColumnar {
    id: u32,
}

#[derive(ToDataFrame)]
#[df_derive(columnar = "columnar_only_runtime::Columnar")]
struct CustomColumnarOnly {
    id: u32,
    #[df_derive(decimal(precision = 18, scale = 2))]
    amount: custom_runtime::CustomDecimal,
}

fn main() {
    let builtin_trait_only = [BuiltinTraitOnly { id: 1 }];
    let df =
        df_derive::dataframe::ToDataFrameVec::to_dataframe(builtin_trait_only.as_slice()).unwrap();
    assert_eq!(df.shape(), (1, 1));

    let builtin_pair = [BuiltinTraitAndColumnar { id: 2 }];
    let df = df_derive::dataframe::ToDataFrameVec::to_dataframe(builtin_pair.as_slice()).unwrap();
    assert_eq!(df.shape(), (1, 1));

    let custom_pair = [CustomTraitAndColumnar { id: 3 }];
    let df = custom_runtime::ToDataFrameVec::to_dataframe(custom_pair.as_slice()).unwrap();
    assert_eq!(df.shape(), (1, 1));

    let custom_columnar_only = CustomColumnarOnly {
        id: 4,
        amount: custom_runtime::CustomDecimal(4250),
    };
    let df = columnar_only_runtime::ToDataFrame::to_dataframe(&custom_columnar_only).unwrap();
    assert_eq!(df.shape(), (1, 2));
}
