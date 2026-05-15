use df_derive::ToDataFrame;
use polars::prelude::{DataFrame, DataType, PolarsResult};

mod custom_runtime {
    use super::*;

    pub trait MyToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
    }

    pub trait MyColumnar: Sized {
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
        T: MyColumnar + MyToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as MyToDataFrame>::empty_dataframe();
            }
            <T as MyColumnar>::columnar_to_dataframe(self)
        }
    }
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

fn main() {
    let builtin_trait_only = [BuiltinTraitOnly { id: 1 }];
    let df = df_derive::dataframe::ToDataFrameVec::to_dataframe(builtin_trait_only.as_slice())
        .unwrap();
    assert_eq!(df.shape(), (1, 1));

    let builtin_pair = [BuiltinTraitAndColumnar { id: 2 }];
    let df = df_derive::dataframe::ToDataFrameVec::to_dataframe(builtin_pair.as_slice()).unwrap();
    assert_eq!(df.shape(), (1, 1));

    let custom_pair = [CustomTraitAndColumnar { id: 3 }];
    let df = custom_runtime::ToDataFrameVec::to_dataframe(custom_pair.as_slice()).unwrap();
    assert_eq!(df.shape(), (1, 1));
}
