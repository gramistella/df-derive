use df_derive::ToDataFrame;
use std::marker::PhantomData;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

#[derive(ToDataFrame)]
struct GenericStruct<T> {
    value: i32,
    _phantom: PhantomData<T>,
}

fn main() {}
