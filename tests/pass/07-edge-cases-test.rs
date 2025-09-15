// Test edge cases to ensure robust Vec<T> implementation

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

// Multiple levels of nesting
#[derive(ToDataFrame)]
struct Coordinates {
    lat: f64,
    lon: f64,
}

#[derive(ToDataFrame)]
struct Location {
    name: String,
    coords: Coordinates,
}

#[derive(ToDataFrame)]
struct Company {
    name: String,
    location: Location,
    employees: u32,
}

// Empty struct in nested position
#[derive(ToDataFrame)]
struct EmptyMeta {}

#[derive(ToDataFrame)]
struct WithEmpty {
    id: u32,
    meta: EmptyMeta,
    value: String,
}

// Optional nested struct
#[derive(ToDataFrame)]
struct OptionalAddress {
    street: String,
    city: String,
}

#[derive(ToDataFrame)]
struct PersonWithOptional {
    name: String,
    address: Option<OptionalAddress>,
}

fn main() {
    // Test 1: Deep nesting (3 levels)
    let companies = vec![
        Company {
            name: "TechCorp".to_string(),
            location: Location {
                name: "Silicon Valley".to_string(),
                coords: Coordinates {
                    lat: 37.3861,
                    lon: -122.0839,
                },
            },
            employees: 500,
        },
        Company {
            name: "StartupInc".to_string(),
            location: Location {
                name: "Austin".to_string(),
                coords: Coordinates {
                    lat: 30.2672,
                    lon: -97.7431,
                },
            },
            employees: 50,
        },
    ];

    let companies_df = companies.to_dataframe().unwrap();

    // Should flatten all levels: name, location.name, location.coords.lat, location.coords.lon, employees
    assert_eq!(companies_df.shape(), (2, 5));

    let expected_deep_columns = [
        "name",
        "location.name",
        "location.coords.lat",
        "location.coords.lon",
        "employees",
    ];

    for expected in &expected_deep_columns {
        assert!(
            companies_df
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Deep nested column '{}' not found",
            expected
        );
    }

    // Test 2: Empty nested struct
    let with_empty = vec![
        WithEmpty {
            id: 1,
            meta: EmptyMeta {},
            value: "test1".to_string(),
        },
        WithEmpty {
            id: 2,
            meta: EmptyMeta {},
            value: "test2".to_string(),
        },
    ];

    let empty_df = with_empty.to_dataframe().unwrap();

    // Empty struct should contribute 0 columns, so only id and value
    assert_eq!(empty_df.shape(), (2, 2));

    let expected_empty_columns = ["id", "value"];
    for expected in &expected_empty_columns {
        assert!(
            empty_df
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Column '{}' not found with empty nested struct",
            expected
        );
    }

    // Test 3: Single item vector (edge case for concatenation)
    let single_company = vec![Company {
        name: "SingleCorp".to_string(),
        location: Location {
            name: "Remote".to_string(),
            coords: Coordinates { lat: 0.0, lon: 0.0 },
        },
        employees: 1,
    }];

    let single_df = single_company.to_dataframe().unwrap();
    assert_eq!(single_df.shape(), (1, 5));

    // Test 4: Optional nested struct with Some and None values
    let people_optional = vec![
        PersonWithOptional {
            name: "Person1".to_string(),
            address: Some(OptionalAddress {
                street: "123 Main St".to_string(),
                city: "CityA".to_string(),
            }),
        },
        PersonWithOptional {
            name: "Person2".to_string(),
            address: None,
        },
    ];

    let optional_df = people_optional.to_dataframe().unwrap();

    // Should have flattened optional nested fields
    assert_eq!(optional_df.shape(), (2, 3));

    let expected_optional_columns = ["name", "address.street", "address.city"];
    for expected in &expected_optional_columns {
        assert!(
            optional_df
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Optional nested column '{}' not found",
            expected
        );
    }

    // Test 5: Empty vector edge case
    let empty_companies: Vec<Company> = vec![];
    let empty_companies_df = empty_companies.to_dataframe().unwrap();

    // Should have same schema as non-empty version but 0 rows
    assert_eq!(empty_companies_df.shape(), (0, 5));
    assert_eq!(
        empty_companies_df.get_column_names(),
        companies_df.get_column_names()
    );

    println!("\n✅ All edge cases passed!");
    println!("   - Deep nesting (3+ levels) works correctly");
    println!("   - Empty nested structs are handled properly");
    println!("   - Single item vectors work (no concatenation needed)");
    println!("   - Optional nested structs flatten correctly");
    println!("   - Empty vectors maintain correct schema");
}
