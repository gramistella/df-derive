use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

#[derive(ToDataFrame)]
struct Comprehensive {
    f64_val: f64,
    i64_val: i64,
    u64_val: u64,
    i32_val: i32,
    u32_val: u32,
    bool_val: bool,
    string_val: String,
    opt_val: Option<i32>,
    vec_val: Vec<String>,
}

fn main() {
    println!("────────────────────────────────────────────────────────");
    println!("🔎 Testing all supported primitive types...");

    let instance = Comprehensive {
        f64_val: 1.0,
        i64_val: -1,
        u64_val: 1,
        i32_val: -2,
        u32_val: 2,
        bool_val: true,
        string_val: "test".to_string(),
        opt_val: Some(42),
        vec_val: vec!["a".to_string(), "b".to_string()],
    };

    let df = instance.to_dataframe().unwrap();
    println!("\n📊 Resulting DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 9));

    let empty_df = Comprehensive::empty_dataframe().unwrap();
    println!("\n📄 Empty DataFrame schema: {:?}", empty_df.get_column_names());
    assert_eq!(empty_df.shape(), (0, 9));

    // Check the schema of the empty DataFrame
    let schema = empty_df.schema();
    assert_eq!(schema.get("f64_val").unwrap(), &DataType::Float64);
    assert_eq!(schema.get("i64_val").unwrap(), &DataType::Int64);
    assert_eq!(schema.get("u64_val").unwrap(), &DataType::UInt64);
    assert_eq!(schema.get("i32_val").unwrap(), &DataType::Int32);
    assert_eq!(schema.get("u32_val").unwrap(), &DataType::UInt32);
    assert_eq!(schema.get("bool_val").unwrap(), &DataType::Boolean);
    assert_eq!(schema.get("string_val").unwrap(), &DataType::String);
    assert_eq!(schema.get("opt_val").unwrap(), &DataType::Int32);
    assert_eq!(
        schema.get("vec_val").unwrap(),
        &DataType::List(Box::new(DataType::String))
    );

    println!("\n✅ All supported types test completed successfully!");
}
