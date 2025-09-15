// This test demonstrates that Vec<T> now properly flattens nested structs
// consistently with the derive macro implementation

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

#[derive(ToDataFrame)]
struct Address {
    street: String,
    city: String,
    zip: u32,
}

#[derive(ToDataFrame)]
struct Person {
    name: String,
    age: u32,
    address: Address,
}

#[derive(ToDataFrame)]
struct Contact {
    email: Option<String>,
    phone: Option<String>,
}

#[derive(ToDataFrame)]
struct Employee {
    id: i64,
    name: String,
    contact: Contact,
}

fn main() {
    // Test 1: Single struct creates flattened columns
    let person = Person {
        name: "John".to_string(),
        age: 30,
        address: Address {
            street: "123 Main St".to_string(),
            city: "Anytown".to_string(),
            zip: 12345,
        },
    };

    let single_df = person.to_dataframe().unwrap();
    println!(
        "Single Person DataFrame columns: {:?}",
        single_df.get_column_names()
    );

    // Test 2: Vec<T> should create the SAME flattened columns
    let people_vec = vec![
        Person {
            name: "Alice".to_string(),
            age: 25,
            address: Address {
                street: "456 Oak Ave".to_string(),
                city: "Other City".to_string(),
                zip: 67890,
            },
        },
        Person {
            name: "Bob".to_string(),
            age: 35,
            address: Address {
                street: "789 Pine St".to_string(),
                city: "Third City".to_string(),
                zip: 54321,
            },
        },
    ];

    let vec_df = people_vec.to_dataframe().unwrap();
    println!(
        "Vec<Person> DataFrame columns: {:?}",
        vec_df.get_column_names()
    );

    // CRITICAL: Vec<T> should have the same column structure as single T
    assert_eq!(single_df.get_column_names(), vec_df.get_column_names());

    // Check that columns are properly flattened (not nested structs)
    let expected_columns = [
        "name",
        "age",
        "address.street",
        "address.city",
        "address.zip",
    ];
    for expected in &expected_columns {
        assert!(
            vec_df
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected flattened column '{}' not found",
            expected
        );
    }

    // Test 3: Test with optional fields in nested structs
    let employees = vec![
        Employee {
            id: 1,
            name: "Employee 1".to_string(),
            contact: Contact {
                email: Some("emp1@example.com".to_string()),
                phone: None,
            },
        },
        Employee {
            id: 2,
            name: "Employee 2".to_string(),
            contact: Contact {
                email: None,
                phone: Some("555-1234".to_string()),
            },
        },
    ];

    let emp_df = employees.to_dataframe().unwrap();
    println!(
        "Vec<Employee> DataFrame columns: {:?}",
        emp_df.get_column_names()
    );

    // Should have flattened optional nested fields
    let expected_emp_columns = ["id", "name", "contact.email", "contact.phone"];
    for expected in &expected_emp_columns {
        assert!(
            emp_df
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected flattened column '{}' not found",
            expected
        );
    }

    // Test 4: Empty vector should have same schema
    let empty_people: Vec<Person> = vec![];
    let empty_df = empty_people.to_dataframe().unwrap();

    assert_eq!(empty_df.get_column_names(), vec_df.get_column_names());
    assert_eq!(empty_df.shape(), (0, 5)); // 0 rows, 5 columns
    assert_eq!(vec_df.shape(), (2, 5)); // 2 rows, 5 columns

    println!("\n✅ SUCCESS: Vec<T> implementation now properly flattens nested structs!");
    println!("   - Vec<T> creates same column structure as single T");
    println!("   - Nested structs are flattened with dot notation (address.street, contact.email)");
    println!("   - Optional nested fields are handled correctly");
    println!("   - Empty vectors maintain correct schema");
}
