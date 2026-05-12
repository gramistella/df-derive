// Additional edge cases for Vec<CustomStruct> functionality

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

// Test 1: Vec<CustomStruct> with Option fields
#[derive(ToDataFrame)]
struct OptionalFields {
    id: u32,
    name: Option<String>,
    value: Option<f64>,
}

#[derive(ToDataFrame)]
struct WithVecOptionals {
    symbol: String,
    data: Vec<OptionalFields>,
}

// Test 2: Empty struct in Vec
#[derive(ToDataFrame)]
struct EmptyStruct {}

#[derive(ToDataFrame)]
struct WithVecEmpty {
    id: String,
    metadata: Vec<EmptyStruct>,
}

// Test 3: Multiple Vec<CustomStruct> fields
#[derive(ToDataFrame)]
struct PriceData {
    price: f64,
    volume: u64,
}

#[derive(ToDataFrame)]
struct VolumeData {
    timestamp: i64,
    amount: f64,
}

#[derive(ToDataFrame)]
struct MultipleVecs {
    symbol: String,
    prices: Vec<PriceData>,
    volumes: Vec<VolumeData>,
}

// Test 4: Option<Vec<CustomStruct>>
#[derive(ToDataFrame)]
struct WithOptionalVec {
    symbol: String,
    optional_data: Option<Vec<PriceData>>,
}

// Test 5: Deeply nested Vec<CustomStruct> (3+ levels)
#[derive(ToDataFrame)]
struct Level3 {
    value: f64,
}

#[derive(ToDataFrame)]
struct Level2 {
    name: String,
    level3: Vec<Level3>,
}

#[derive(ToDataFrame)]
struct Level1 {
    id: u32,
    level2: Vec<Level2>,
}

fn main() {
    test_vec_with_optional_fields();
    test_vec_with_empty_struct();
    test_multiple_vec_custom_struct_fields();
    test_optional_vec_custom_struct();
    test_deeply_nested_vec_custom_struct();
}

fn test_vec_with_optional_fields() {
    let data = WithVecOptionals {
        symbol: "TEST".to_string(),
        data: vec![
            OptionalFields {
                id: 1,
                name: Some("First".to_string()),
                value: Some(10.5),
            },
            OptionalFields {
                id: 2,
                name: None,
                value: Some(20.3),
            },
            OptionalFields {
                id: 3,
                name: Some("Third".to_string()),
                value: None,
            },
        ],
    };

    let df = data.to_dataframe().unwrap();

    // Should have: symbol, data.id, data.name, data.value
    assert_eq!(df.shape(), (1, 4));

    let expected_columns = ["symbol", "data.id", "data.name", "data.value"];
    for expected in &expected_columns {
        assert!(
            df.get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    println!("✅ Vec<CustomStruct> with optional fields test passed!");
}

fn test_vec_with_empty_struct() {
    let data = WithVecEmpty {
        id: "empty_test".to_string(),
        metadata: vec![EmptyStruct {}, EmptyStruct {}, EmptyStruct {}],
    };

    let df = data.to_dataframe().unwrap();

    // Should only have the id column since EmptyStruct contributes no columns
    assert_eq!(df.shape(), (1, 1));
    assert_eq!(df.get_column_names(), vec!["id"]);

    println!("✅ Vec<EmptyStruct> test passed!");
}

fn test_multiple_vec_custom_struct_fields() {
    let data = MultipleVecs {
        symbol: "MULTI".to_string(),
        prices: vec![
            PriceData {
                price: 100.0,
                volume: 1000,
            },
            PriceData {
                price: 101.0,
                volume: 1500,
            },
        ],
        volumes: vec![
            VolumeData {
                timestamp: 1234567890,
                amount: 50.0,
            },
            VolumeData {
                timestamp: 1234567891,
                amount: 75.0,
            },
        ],
    };

    let df = data.to_dataframe().unwrap();

    // Should have: symbol, prices.price, prices.volume, volumes.timestamp, volumes.amount
    assert_eq!(df.shape(), (1, 5));

    let expected_columns = [
        "symbol",
        "prices.price",
        "prices.volume",
        "volumes.timestamp",
        "volumes.amount",
    ];

    for expected in &expected_columns {
        assert!(
            df.get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    println!("✅ Multiple Vec<CustomStruct> fields test passed!");
}

fn test_optional_vec_custom_struct() {
    // Test with Some(Vec<...>)
    let data_with_vec = WithOptionalVec {
        symbol: "WITH_VEC".to_string(),
        optional_data: Some(vec![
            PriceData {
                price: 200.0,
                volume: 2000,
            },
            PriceData {
                price: 201.0,
                volume: 2100,
            },
        ]),
    };

    let df_with = data_with_vec.to_dataframe().unwrap();

    // Should have: symbol, optional_data.price, optional_data.volume
    assert_eq!(df_with.shape(), (1, 3));

    let expected_columns = ["symbol", "optional_data.price", "optional_data.volume"];
    for expected in &expected_columns {
        assert!(
            df_with
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    // Test with None
    let data_without_vec = WithOptionalVec {
        symbol: "WITHOUT_VEC".to_string(),
        optional_data: None,
    };

    let df_without = data_without_vec.to_dataframe().unwrap();

    // Should have same schema but with null values
    assert_eq!(df_without.shape(), (1, 3));
    assert_eq!(df_without.get_column_names(), df_with.get_column_names());

    println!("✅ Option<Vec<CustomStruct>> test passed!");
}

fn test_deeply_nested_vec_custom_struct() {
    let data = Level1 {
        id: 1,
        level2: vec![
            Level2 {
                name: "First".to_string(),
                level3: vec![Level3 { value: 1.1 }, Level3 { value: 1.2 }],
            },
            Level2 {
                name: "Second".to_string(),
                level3: vec![Level3 { value: 2.1 }],
            },
        ],
    };

    let df = data.to_dataframe().unwrap();

    // Should have: id, level2.name, level2.level3.value
    assert_eq!(df.shape(), (1, 3));

    let expected_columns = ["id", "level2.name", "level2.level3.value"];
    for expected in &expected_columns {
        assert!(
            df.get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    println!("✅ Deeply nested Vec<CustomStruct> test passed!");
}
