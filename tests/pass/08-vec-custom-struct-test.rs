// Test the Vec<CustomStruct> issue described in the problem

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

// This is the failing example from the problem description
#[derive(ToDataFrame)]
struct OptionContract {
    strike: f64,
    last_price: f64,
}

#[derive(ToDataFrame)] // <-- This derive will fail
struct OptionChain {
    underlying_symbol: String,
    expiration_date: i64,
    calls: Vec<OptionContract>, // <-- PROBLEM: This is a Vec of another struct
}

fn main() {
    test_vec_custom_struct();
    test_empty_vec_custom_struct();
    test_nested_vec_custom_struct();
    test_mixed_fields_with_vec_custom_struct();
}

fn test_vec_custom_struct() {
    // This should work: create an OptionChain with Vec<OptionContract>
    let option_chain = OptionChain {
        underlying_symbol: "AAPL".to_string(),
        expiration_date: 1755494400,
        calls: vec![
            OptionContract {
                strike: 180.0,
                last_price: 25.5,
            },
            OptionContract {
                strike: 185.0,
                last_price: 21.2,
            },
        ],
    };

    // Convert to DataFrame
    let df = option_chain.to_dataframe().unwrap();

    // Expected behavior: single-row DataFrame with List columns
    // underlying_symbol | expiration_date | calls.strike         | calls.last_price
    // "AAPL"           | 1755494400      | [180.0, 185.0, ...] | [25.5, 21.2, ...]

    println!("OptionChain DataFrame:");
    println!("{}", df);

    // Check the shape - should be 1 row with 4 columns
    assert_eq!(df.shape(), (1, 4));

    // Check column names
    let expected_columns = [
        "underlying_symbol",
        "expiration_date",
        "calls.strike",
        "calls.last_price",
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

    // Verify the data types and values - List columns should contain correct data
    use polars::prelude::{AnyValue, Column};

    // Helper function to extract Vec<T> from a List Series
    fn extract_list_series_as_vec<T: polars::prelude::PolarsNumericType>(
        series: &Column,
    ) -> Vec<T::Native>
    where
        T::Native: Copy,
    {
        if let AnyValue::List(s) = series.get(0).unwrap() {
            let ca = s.unpack::<T>().unwrap();
            return ca.into_no_null_iter().collect();
        }
        panic!("Expected a List AnyValue");
    }

    let calls_strike_vec: Vec<f64> = extract_list_series_as_vec::<polars::prelude::Float64Type>(
        df.column("calls.strike").unwrap(),
    );
    let calls_price_vec: Vec<f64> = extract_list_series_as_vec::<polars::prelude::Float64Type>(
        df.column("calls.last_price").unwrap(),
    );

    assert_eq!(calls_strike_vec, vec![180.0, 185.0]);
    assert_eq!(calls_price_vec, vec![25.5, 21.2]);

    println!("\n✅ SUCCESS: Vec<CustomStruct> fields now work correctly!");
    println!("   - Fields are flattened with dot notation (calls.strike, calls.last_price)");
    println!("   - Each flattened field contains a List of values from the Vec elements");
    println!("   - Assertions now validate the inner data of the List Series");
}

// Test with empty Vec<CustomStruct>
fn test_empty_vec_custom_struct() {
    let empty_chain = OptionChain {
        underlying_symbol: "SPY".to_string(),
        expiration_date: 1755494400,
        calls: vec![], // Empty vec
    };

    let df = empty_chain.to_dataframe().unwrap();

    // Should still have the same column structure but with empty lists
    assert_eq!(df.shape(), (1, 4));

    let expected_columns = [
        "underlying_symbol",
        "expiration_date",
        "calls.strike",
        "calls.last_price",
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

    println!("✅ Empty Vec<CustomStruct> test passed!");
}

// Test with nested struct containing Vec<CustomStruct>
#[derive(ToDataFrame)]
struct NestedOptionData {
    symbol: String,
    chains: Vec<OptionChain>,
}

fn test_nested_vec_custom_struct() {
    let nested_data = NestedOptionData {
        symbol: "TSLA".to_string(),
        chains: vec![OptionChain {
            underlying_symbol: "TSLA".to_string(),
            expiration_date: 1755494400,
            calls: vec![
                OptionContract {
                    strike: 200.0,
                    last_price: 50.0,
                },
                OptionContract {
                    strike: 210.0,
                    last_price: 45.0,
                },
            ],
        }],
    };

    let df = nested_data.to_dataframe().unwrap();

    // Should have flattened columns: symbol, chains.underlying_symbol, chains.expiration_date, chains.calls.strike, chains.calls.last_price
    assert_eq!(df.shape(), (1, 5));

    let expected_columns = [
        "symbol",
        "chains.underlying_symbol",
        "chains.expiration_date",
        "chains.calls.strike",
        "chains.calls.last_price",
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

    println!("✅ Nested Vec<CustomStruct> test passed!");
}

// Test mixed primitive and Vec<CustomStruct> fields
#[derive(ToDataFrame)]
struct MixedData {
    id: u32,
    name: String,
    contracts: Vec<OptionContract>,
    active: bool,
}

fn test_mixed_fields_with_vec_custom_struct() {
    let mixed = MixedData {
        id: 123,
        name: "Test Portfolio".to_string(),
        contracts: vec![
            OptionContract {
                strike: 100.0,
                last_price: 5.0,
            },
            OptionContract {
                strike: 105.0,
                last_price: 3.0,
            },
            OptionContract {
                strike: 110.0,
                last_price: 1.0,
            },
        ],
        active: true,
    };

    let df = mixed.to_dataframe().unwrap();

    // Should have: id, name, contracts.strike, contracts.last_price, active
    assert_eq!(df.shape(), (1, 5));

    let expected_columns = [
        "id",
        "name",
        "contracts.strike",
        "contracts.last_price",
        "active",
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

    println!("✅ Mixed fields with Vec<CustomStruct> test passed!");
}
