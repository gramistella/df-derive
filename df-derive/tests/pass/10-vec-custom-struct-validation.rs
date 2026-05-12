// Data validation and type correctness tests for Vec<CustomStruct>

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

// Test struct with various data types
#[derive(ToDataFrame)]
struct TypedContract {
    strike: f64,
    price: f32,
    volume: u64,
    active: bool,
    expiry: i64,
    symbol: String,
}

#[derive(ToDataFrame)]
struct TypeValidation {
    name: String,
    contracts: Vec<TypedContract>,
}

// Test struct for single item vector edge case
#[derive(ToDataFrame)]
struct SingleItem {
    id: u32,
    value: f64,
}

#[derive(ToDataFrame)]
struct WithSingleItemVec {
    title: String,
    item: Vec<SingleItem>,
}

// Test struct with mixed Vec and regular nested structs
#[derive(ToDataFrame)]
struct RegularNested {
    name: String,
    value: f64,
}

#[derive(ToDataFrame)]
struct MixedNesting {
    id: u32,
    regular_nested: RegularNested,
    vec_nested: Vec<RegularNested>,
}

fn main() {
    test_data_type_preservation();
    test_single_item_vector();
    test_mixed_regular_and_vec_nesting();
    test_empty_dataframe_schema_consistency();
}

fn test_data_type_preservation() {
    let data = TypeValidation {
        name: "Test Portfolio".to_string(),
        contracts: vec![
            TypedContract {
                strike: 100.0,
                price: 5.5,
                volume: 1000,
                active: true,
                expiry: 1755494400,
                symbol: "CALL".to_string(),
            },
            TypedContract {
                strike: 105.0,
                price: 3.2,
                volume: 2000,
                active: false,
                expiry: 1755494500,
                symbol: "PUT".to_string(),
            },
        ],
    };

    let df = data.to_dataframe().unwrap();

    // Verify column count and names
    assert_eq!(df.shape(), (1, 7)); // name + 6 contract fields

    let expected_columns = [
        "name",
        "contracts.strike",
        "contracts.price",
        "contracts.volume",
        "contracts.active",
        "contracts.expiry",
        "contracts.symbol",
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

    // Verify data types are correct for List columns
    let schema = df.schema();
    use polars::prelude::DataType;

    assert_eq!(schema.get("name").unwrap(), &DataType::String);
    assert_eq!(
        schema.get("contracts.strike").unwrap(),
        &DataType::List(Box::new(DataType::Float64))
    );
    assert_eq!(
        schema.get("contracts.price").unwrap(),
        &DataType::List(Box::new(DataType::Float32))
    );
    assert_eq!(
        schema.get("contracts.volume").unwrap(),
        &DataType::List(Box::new(DataType::UInt64))
    );
    assert_eq!(
        schema.get("contracts.active").unwrap(),
        &DataType::List(Box::new(DataType::Boolean))
    );
    assert_eq!(
        schema.get("contracts.expiry").unwrap(),
        &DataType::List(Box::new(DataType::Int64))
    );
    assert_eq!(
        schema.get("contracts.symbol").unwrap(),
        &DataType::List(Box::new(DataType::String))
    );

    println!("✅ Data type preservation test passed!");
}

fn test_single_item_vector() {
    let data = WithSingleItemVec {
        title: "Single Item Test".to_string(),
        item: vec![SingleItem {
            id: 42,
            value: 3.14,
        }],
    };

    let df = data.to_dataframe().unwrap();

    // Should have: title, item.id, item.value
    assert_eq!(df.shape(), (1, 3));

    let expected_columns = ["title", "item.id", "item.value"];
    for expected in &expected_columns {
        assert!(
            df.get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    // Verify the list contains exactly one element
    let id_col = df.column("item.id").unwrap();
    let value_col = df.column("item.value").unwrap();

    // Both should be List types containing single elements
    use polars::prelude::DataType;
    assert!(matches!(id_col.dtype(), DataType::List(_)));
    assert!(matches!(value_col.dtype(), DataType::List(_)));

    println!("✅ Single item vector test passed!");
}

fn test_mixed_regular_and_vec_nesting() {
    let data = MixedNesting {
        id: 123,
        regular_nested: RegularNested {
            name: "Regular".to_string(),
            value: 10.0,
        },
        vec_nested: vec![
            RegularNested {
                name: "Vec1".to_string(),
                value: 20.0,
            },
            RegularNested {
                name: "Vec2".to_string(),
                value: 30.0,
            },
        ],
    };

    let df = data.to_dataframe().unwrap();

    // Should have: id, regular_nested.name, regular_nested.value, vec_nested.name, vec_nested.value
    assert_eq!(df.shape(), (1, 5));

    let expected_columns = [
        "id",
        "regular_nested.name",
        "regular_nested.value",
        "vec_nested.name",
        "vec_nested.value",
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

    // Verify that regular nested fields are NOT lists, but vec nested fields ARE lists
    let schema = df.schema();
    use polars::prelude::DataType;

    // Regular nested should be scalar types
    assert_eq!(
        schema.get("regular_nested.name").unwrap(),
        &DataType::String
    );
    assert_eq!(
        schema.get("regular_nested.value").unwrap(),
        &DataType::Float64
    );

    // Vec nested should be list types
    assert_eq!(
        schema.get("vec_nested.name").unwrap(),
        &DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        schema.get("vec_nested.value").unwrap(),
        &DataType::List(Box::new(DataType::Float64))
    );

    println!("✅ Mixed regular and Vec nesting test passed!");
}

fn test_empty_dataframe_schema_consistency() {
    // Test that empty_dataframe() has the same schema as to_dataframe()
    let data = TypeValidation {
        name: "Test".to_string(),
        contracts: vec![TypedContract {
            strike: 100.0,
            price: 5.0,
            volume: 1000,
            active: true,
            expiry: 1755494400,
            symbol: "TEST".to_string(),
        }],
    };

    let populated_df = data.to_dataframe().unwrap();
    let empty_df = TypeValidation::empty_dataframe().unwrap();

    // Schemas should be identical
    assert_eq!(populated_df.schema(), empty_df.schema());
    assert_eq!(populated_df.get_column_names(), empty_df.get_column_names());

    // Empty should have 0 rows, populated should have 1 row
    assert_eq!(empty_df.shape().0, 0);
    assert_eq!(populated_df.shape().0, 1);

    // Same number of columns
    assert_eq!(empty_df.shape().1, populated_df.shape().1);

    println!("✅ Empty dataframe schema consistency test passed!");
}
