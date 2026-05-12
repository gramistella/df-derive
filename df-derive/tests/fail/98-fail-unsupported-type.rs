use df_derive::ToDataFrame;
use std::collections::HashMap;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Unsupported {
    id: i32,
    // HashMap is not a supported type for conversion to a Polars Series
    metadata: HashMap<String, String>,
}

fn main() {}
