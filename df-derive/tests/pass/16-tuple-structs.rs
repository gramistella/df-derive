use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

// Simple tuple struct with basic types
#[derive(ToDataFrame)]
struct SimpleTuple(i32, String, f64);

// Tuple struct with optional types
#[derive(ToDataFrame)]
struct OptionalTuple(Option<i32>, Option<String>, bool);

// Tuple struct with vector types
#[derive(ToDataFrame)]
struct VectorTuple(Vec<i32>, Vec<String>, u64);

// Mixed tuple struct with various types
#[derive(ToDataFrame)]
struct MixedTuple(i64, Option<f64>, Vec<bool>, String, u32);

// Empty tuple struct (unit-like)
#[derive(ToDataFrame)]
struct EmptyTuple();

// Single field tuple struct
#[derive(ToDataFrame)]
struct SingleField(String);

// Large tuple struct to test many fields
#[derive(ToDataFrame)]
struct LargeTuple(i32, i64, f32, f64, bool, String, Option<i32>, Vec<String>);

fn main() {
    println!("────────────────────────────────────────────────────────");
    println!("🔎 Testing tuple structs → DataFrame conversion...");

    // Test 1: Simple tuple struct
    test_simple_tuple();

    // Test 2: Optional tuple struct
    test_optional_tuple();

    // Test 3: Vector tuple struct
    test_vector_tuple();

    // Test 4: Mixed tuple struct
    test_mixed_tuple();

    // Test 5: Empty tuple struct
    test_empty_tuple();

    // Test 6: Single field tuple struct
    test_single_field();

    // Test 7: Large tuple struct
    test_large_tuple();

    // Test 8: Empty dataframes
    test_empty_dataframes();

    println!("\n✅ All tuple struct tests completed successfully!");
}

fn test_simple_tuple() {
    println!("\n📋 Testing simple tuple struct...");

    let simple = SimpleTuple(42, "Hello World".to_string(), 3.14159);
    let df = simple.to_dataframe().unwrap();
    
    println!("\n📊 SimpleTuple DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 3));

    // Check column names (should be field_0, field_1, field_2)
    let column_names = df.get_column_names();
    let expected_columns = ["field_0", "field_1", "field_2"];
    assert_eq!(column_names, expected_columns);

    // Check data types
    assert_eq!(df.column("field_0").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("field_1").unwrap().dtype(), &DataType::String);
    assert_eq!(df.column("field_2").unwrap().dtype(), &DataType::Float64);

    // Check values
    assert_eq!(df.column("field_0").unwrap().get(0).unwrap(), AnyValue::Int32(42));
    assert_eq!(df.column("field_1").unwrap().get(0).unwrap(), AnyValue::String("Hello World"));
    assert_eq!(df.column("field_2").unwrap().get(0).unwrap(), AnyValue::Float64(3.14159));

    println!("✅ Simple tuple struct test passed!");
}

fn test_optional_tuple() {
    println!("\n📋 Testing optional tuple struct...");

    let optional = OptionalTuple(Some(100), None, true);
    let df = optional.to_dataframe().unwrap();
    
    println!("\n📊 OptionalTuple DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 3));

    let column_names = df.get_column_names();
    let expected_columns = ["field_0", "field_1", "field_2"];
    assert_eq!(column_names, expected_columns);

    // Check data types
    assert_eq!(df.column("field_0").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("field_1").unwrap().dtype(), &DataType::String);
    assert_eq!(df.column("field_2").unwrap().dtype(), &DataType::Boolean);

    // Check values
    assert_eq!(df.column("field_0").unwrap().get(0).unwrap(), AnyValue::Int32(100));
    assert!(matches!(df.column("field_1").unwrap().get(0).unwrap(), AnyValue::Null));
    assert_eq!(df.column("field_2").unwrap().get(0).unwrap(), AnyValue::Boolean(true));

    println!("✅ Optional tuple struct test passed!");
}

fn test_vector_tuple() {
    println!("\n📋 Testing vector tuple struct...");

    let vector = VectorTuple(
        vec![1, 2, 3, 4, 5],
        vec!["apple".to_string(), "banana".to_string(), "cherry".to_string()],
        999
    );
    let df = vector.to_dataframe().unwrap();
    
    println!("\n📊 VectorTuple DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 3));

    let column_names = df.get_column_names();
    let expected_columns = ["field_0", "field_1", "field_2"];
    assert_eq!(column_names, expected_columns);

    // Check data types
    assert_eq!(df.column("field_0").unwrap().dtype(), &DataType::List(Box::new(DataType::Int32)));
    assert_eq!(df.column("field_1").unwrap().dtype(), &DataType::List(Box::new(DataType::String)));
    assert_eq!(df.column("field_2").unwrap().dtype(), &DataType::UInt64);

    // Check values
    assert!(matches!(df.column("field_0").unwrap().get(0).unwrap(), AnyValue::List(_)));
    assert!(matches!(df.column("field_1").unwrap().get(0).unwrap(), AnyValue::List(_)));
    assert_eq!(df.column("field_2").unwrap().get(0).unwrap(), AnyValue::UInt64(999));

    println!("✅ Vector tuple struct test passed!");
}

fn test_mixed_tuple() {
    println!("\n📋 Testing mixed tuple struct...");

    let mixed = MixedTuple(
        -12345,
        Some(2.71828),
        vec![true, false, true, false],
        "Mixed Types".to_string(),
        42
    );
    let df = mixed.to_dataframe().unwrap();
    
    println!("\n📊 MixedTuple DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 5));

    let column_names = df.get_column_names();
    let expected_columns = ["field_0", "field_1", "field_2", "field_3", "field_4"];
    assert_eq!(column_names, expected_columns);

    // Check data types
    assert_eq!(df.column("field_0").unwrap().dtype(), &DataType::Int64);
    assert_eq!(df.column("field_1").unwrap().dtype(), &DataType::Float64);
    assert_eq!(df.column("field_2").unwrap().dtype(), &DataType::List(Box::new(DataType::Boolean)));
    assert_eq!(df.column("field_3").unwrap().dtype(), &DataType::String);
    assert_eq!(df.column("field_4").unwrap().dtype(), &DataType::UInt32);

    // Check values
    assert_eq!(df.column("field_0").unwrap().get(0).unwrap(), AnyValue::Int64(-12345));
    assert_eq!(df.column("field_1").unwrap().get(0).unwrap(), AnyValue::Float64(2.71828));
    assert!(matches!(df.column("field_2").unwrap().get(0).unwrap(), AnyValue::List(_)));
    assert_eq!(df.column("field_3").unwrap().get(0).unwrap(), AnyValue::String("Mixed Types"));
    assert_eq!(df.column("field_4").unwrap().get(0).unwrap(), AnyValue::UInt32(42));

    println!("✅ Mixed tuple struct test passed!");
}

fn test_empty_tuple() {
    println!("\n📋 Testing empty tuple struct...");

    let empty = EmptyTuple();
    let df = empty.to_dataframe().unwrap();
    
    println!("\n📊 EmptyTuple DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 0));

    let column_names = df.get_column_names();
    assert_eq!(column_names.len(), 0);

    println!("✅ Empty tuple struct test passed!");
}

fn test_single_field() {
    println!("\n📋 Testing single field tuple struct...");

    let single = SingleField("Only Field".to_string());
    let df = single.to_dataframe().unwrap();
    
    println!("\n📊 SingleField DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 1));

    let column_names = df.get_column_names();
    let expected_columns = ["field_0"];
    assert_eq!(column_names, expected_columns);

    // Check data type
    assert_eq!(df.column("field_0").unwrap().dtype(), &DataType::String);

    // Check value
    assert_eq!(df.column("field_0").unwrap().get(0).unwrap(), AnyValue::String("Only Field"));

    println!("✅ Single field tuple struct test passed!");
}

fn test_large_tuple() {
    println!("\n📋 Testing large tuple struct...");

    let large = LargeTuple(
        123,
        -456789,
        1.5,
        2.71828,
        true,
        "Large Tuple".to_string(),
        Some(999),
        vec!["item1".to_string(), "item2".to_string(), "item3".to_string()]
    );
    let df = large.to_dataframe().unwrap();
    
    println!("\n📊 LargeTuple DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 8));

    let column_names = df.get_column_names();
    let expected_columns = ["field_0", "field_1", "field_2", "field_3", "field_4", "field_5", "field_6", "field_7"];
    assert_eq!(column_names, expected_columns);

    // Check data types
    assert_eq!(df.column("field_0").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("field_1").unwrap().dtype(), &DataType::Int64);
    assert_eq!(df.column("field_2").unwrap().dtype(), &DataType::Float32);
    assert_eq!(df.column("field_3").unwrap().dtype(), &DataType::Float64);
    assert_eq!(df.column("field_4").unwrap().dtype(), &DataType::Boolean);
    assert_eq!(df.column("field_5").unwrap().dtype(), &DataType::String);
    assert_eq!(df.column("field_6").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("field_7").unwrap().dtype(), &DataType::List(Box::new(DataType::String)));

    // Check values
    assert_eq!(df.column("field_0").unwrap().get(0).unwrap(), AnyValue::Int32(123));
    assert_eq!(df.column("field_1").unwrap().get(0).unwrap(), AnyValue::Int64(-456789));
    assert_eq!(df.column("field_2").unwrap().get(0).unwrap(), AnyValue::Float32(1.5));
    assert_eq!(df.column("field_3").unwrap().get(0).unwrap(), AnyValue::Float64(2.71828));
    assert_eq!(df.column("field_4").unwrap().get(0).unwrap(), AnyValue::Boolean(true));
    assert_eq!(df.column("field_5").unwrap().get(0).unwrap(), AnyValue::String("Large Tuple"));
    assert_eq!(df.column("field_6").unwrap().get(0).unwrap(), AnyValue::Int32(999));
    assert!(matches!(df.column("field_7").unwrap().get(0).unwrap(), AnyValue::List(_)));

    println!("✅ Large tuple struct test passed!");
}

fn test_empty_dataframes() {
    println!("\n📋 Testing empty dataframes for tuple structs...");

    // Test empty dataframe for simple tuple
    let simple_empty = SimpleTuple::empty_dataframe().unwrap();
    println!("\n📄 SimpleTuple empty DataFrame columns: {:?}", simple_empty.get_column_names());
    assert_eq!(simple_empty.shape(), (0, 3));
    assert_eq!(simple_empty.get_column_names(), &["field_0", "field_1", "field_2"]);

    // Test empty dataframe for optional tuple
    let optional_empty = OptionalTuple::empty_dataframe().unwrap();
    assert_eq!(optional_empty.shape(), (0, 3));
    assert_eq!(optional_empty.get_column_names(), &["field_0", "field_1", "field_2"]);

    // Test empty dataframe for vector tuple
    let vector_empty = VectorTuple::empty_dataframe().unwrap();
    assert_eq!(vector_empty.shape(), (0, 3));
    assert_eq!(vector_empty.get_column_names(), &["field_0", "field_1", "field_2"]);

    // Test empty dataframe for mixed tuple
    let mixed_empty = MixedTuple::empty_dataframe().unwrap();
    assert_eq!(mixed_empty.shape(), (0, 5));
    assert_eq!(mixed_empty.get_column_names(), &["field_0", "field_1", "field_2", "field_3", "field_4"]);

    // Test empty dataframe for empty tuple
    let empty_empty = EmptyTuple::empty_dataframe().unwrap();
    assert_eq!(empty_empty.shape(), (0, 0));
    assert_eq!(empty_empty.get_column_names().len(), 0);

    // Test empty dataframe for single field tuple
    let single_empty = SingleField::empty_dataframe().unwrap();
    assert_eq!(single_empty.shape(), (0, 1));
    assert_eq!(single_empty.get_column_names(), &["field_0"]);

    // Test empty dataframe for large tuple
    let large_empty = LargeTuple::empty_dataframe().unwrap();
    assert_eq!(large_empty.shape(), (0, 8));
    assert_eq!(large_empty.get_column_names(), &["field_0", "field_1", "field_2", "field_3", "field_4", "field_5", "field_6", "field_7"]);

    println!("✅ Empty dataframes test passed!");
}
