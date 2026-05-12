use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

// Basic nested struct
#[derive(ToDataFrame)]
struct Address {
    street: String,
    city: String,
    zip: u32,
}

// Multiple data types and nested struct
#[derive(ToDataFrame)]
struct Person {
    name: String,
    age: u32,
    address: Address,
    salary: f64,
}

// Nested struct with optional fields
#[derive(ToDataFrame)]
struct Contact {
    email: Option<String>,
    phone: Option<String>,
    verified: bool,
}

// Complex nested structure with multiple nested structs
#[derive(ToDataFrame)]
struct Employee {
    id: i64,
    name: String,
    address: Address,
    contact: Contact,
    salary: f64,
    active: bool,
}

// Multiple levels of nesting
#[derive(ToDataFrame)]
struct Coordinates {
    latitude: f64,
    longitude: f64,
}

#[derive(ToDataFrame)]
struct Location {
    name: String,
    coordinates: Coordinates,
}

#[derive(ToDataFrame)]
struct Company {
    name: String,
    location: Location,
    employee_count: u32,
}

// Nested struct with Vec fields
#[derive(ToDataFrame)]
struct Skills {
    programming: Vec<String>,
    languages: Vec<String>,
}

#[derive(ToDataFrame)]
struct Developer {
    name: String,
    skills: Skills,
    experience_years: u32,
}

// Empty nested struct
#[derive(ToDataFrame)]
struct EmptyNested {}

#[derive(ToDataFrame)]
struct WithEmpty {
    id: u32,
    empty: EmptyNested,
    value: String,
}

// Multiple instances of the same nested type
#[derive(ToDataFrame)]
struct MultipleAddresses {
    id: u32,
    home_address: Address,
    work_address: Address,
    name: String,
}

// Mixed nesting: primitives, nested structs, and options
#[derive(ToDataFrame)]
struct ComplexMixed {
    id: i64,
    primary_contact: Option<Contact>,
    backup_contact: Contact,
    addresses: Vec<String>, // Simple vector
    main_location: Location,
    active: bool,
}

fn main() {
    // Test 1: Basic nested struct functionality
    test_basic_nested_struct();

    // Test 2: Complex nested structure with multiple nested structs
    test_complex_nested_structure();

    // Test 3: Multiple levels of nesting (3 levels deep)
    test_deep_nesting();

    // Test 4: Nested struct with Vec fields
    test_nested_with_vectors();

    // Test 5: Empty nested struct
    test_empty_nested_struct();

    // Test 6: Empty dataframe functionality
    test_empty_dataframes();

    // Test 7: Data types and values validation
    test_data_types_and_values();

    // Test 8: Multiple instances of same nested type
    test_multiple_nested_instances();

    // Test 9: Complex mixed scenarios
    test_complex_mixed_scenarios();

    println!("All nested struct tests passed!");
}

fn test_basic_nested_struct() {
    println!("Testing basic nested struct...");

    let person = Person {
        name: "John Doe".to_string(),
        age: 30,
        address: Address {
            street: "123 Main St".to_string(),
            city: "Anytown".to_string(),
            zip: 12345,
        },
        salary: 75000.0,
    };

    let df = person.to_dataframe().unwrap();

    // Should flatten nested struct fields with dot notation
    // Expected columns: name, age, address.street, address.city, address.zip, salary
    assert_eq!(df.shape(), (1, 6));

    let column_names = df.get_column_names();
    let expected_columns = [
        "name",
        "age",
        "address.street",
        "address.city",
        "address.zip",
        "salary",
    ];
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in {:?}",
            expected_col,
            column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    // Test specific values
    assert_eq!(
        df.column("name").unwrap().get(0).unwrap(),
        AnyValue::String("John Doe")
    );
    assert_eq!(
        df.column("age").unwrap().get(0).unwrap(),
        AnyValue::UInt32(30)
    );
    assert_eq!(
        df.column("address.street").unwrap().get(0).unwrap(),
        AnyValue::String("123 Main St")
    );
    assert_eq!(
        df.column("address.city").unwrap().get(0).unwrap(),
        AnyValue::String("Anytown")
    );
    assert_eq!(
        df.column("address.zip").unwrap().get(0).unwrap(),
        AnyValue::UInt32(12345)
    );
    assert_eq!(
        df.column("salary").unwrap().get(0).unwrap(),
        AnyValue::Float64(75000.0)
    );
}

fn test_complex_nested_structure() {
    println!("Testing complex nested structure...");

    let employee = Employee {
        id: 12345,
        name: "Jane Smith".to_string(),
        address: Address {
            street: "456 Oak Ave".to_string(),
            city: "Springfield".to_string(),
            zip: 67890,
        },
        contact: Contact {
            email: Some("jane.smith@example.com".to_string()),
            phone: Some("555-1234".to_string()),
            verified: true,
        },
        salary: 85000.0,
        active: true,
    };

    let df = employee.to_dataframe().unwrap();

    // Expected columns: id, name, address.*, contact.*, salary, active
    // Adjust the count based on what we expect:
    // id(1) + name(1) + address(3) + contact(3) + salary(1) + active(1) = 10 columns
    assert_eq!(df.shape(), (1, 10));

    let expected_columns = [
        "id",
        "name",
        "address.street",
        "address.city",
        "address.zip",
        "contact.email",
        "contact.phone",
        "contact.verified",
        "salary",
        "active",
    ];

    let column_names = df.get_column_names();
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in {:?}",
            expected_col,
            column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    // Test optional field values
    assert_eq!(
        df.column("contact.email").unwrap().get(0).unwrap(),
        AnyValue::String("jane.smith@example.com")
    );
    assert_eq!(
        df.column("contact.phone").unwrap().get(0).unwrap(),
        AnyValue::String("555-1234")
    );
    assert_eq!(
        df.column("contact.verified").unwrap().get(0).unwrap(),
        AnyValue::Boolean(true)
    );
}

fn test_deep_nesting() {
    println!("Testing deep nesting (3 levels)...");

    let company = Company {
        name: "Tech Corp".to_string(),
        location: Location {
            name: "Silicon Valley".to_string(),
            coordinates: Coordinates {
                latitude: 37.3861,
                longitude: -122.0839,
            },
        },
        employee_count: 500,
    };

    let df = company.to_dataframe().unwrap();

    // Expected columns: name, location.name, location.coordinates.latitude, location.coordinates.longitude, employee_count
    assert_eq!(df.shape(), (1, 5));

    let expected_columns = [
        "name",
        "location.name",
        "location.coordinates.latitude",
        "location.coordinates.longitude",
        "employee_count",
    ];

    let column_names = df.get_column_names();
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in {:?}",
            expected_col,
            column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    // Test deep nested values
    assert_eq!(
        df.column("name").unwrap().get(0).unwrap(),
        AnyValue::String("Tech Corp")
    );
    assert_eq!(
        df.column("location.name").unwrap().get(0).unwrap(),
        AnyValue::String("Silicon Valley")
    );
    assert_eq!(
        df.column("location.coordinates.latitude")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Float64(37.3861)
    );
    assert_eq!(
        df.column("location.coordinates.longitude")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Float64(-122.0839)
    );
    assert_eq!(
        df.column("employee_count").unwrap().get(0).unwrap(),
        AnyValue::UInt32(500)
    );
}

fn test_nested_with_vectors() {
    println!("Testing nested struct with vectors...");

    let developer = Developer {
        name: "Alice Johnson".to_string(),
        skills: Skills {
            programming: vec![
                "Rust".to_string(),
                "Python".to_string(),
                "JavaScript".to_string(),
            ],
            languages: vec!["English".to_string(), "Spanish".to_string()],
        },
        experience_years: 5,
    };

    let df = developer.to_dataframe().unwrap();

    // Expected columns: name, skills.programming, skills.languages, experience_years
    assert_eq!(df.shape(), (1, 4));

    let expected_columns = [
        "name",
        "skills.programming",
        "skills.languages",
        "experience_years",
    ];

    let column_names = df.get_column_names();
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in {:?}",
            expected_col,
            column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    // Test that vector fields are properly handled
    assert_eq!(
        df.column("name").unwrap().get(0).unwrap(),
        AnyValue::String("Alice Johnson")
    );
    assert_eq!(
        df.column("experience_years").unwrap().get(0).unwrap(),
        AnyValue::UInt32(5)
    );

    // Vector fields should be List types
    let programming_value = df.column("skills.programming").unwrap().get(0).unwrap();
    let languages_value = df.column("skills.languages").unwrap().get(0).unwrap();

    // Check that these are list types (exact value checking for lists is complex)
    assert!(matches!(programming_value, AnyValue::List(_)));
    assert!(matches!(languages_value, AnyValue::List(_)));
}

fn test_empty_nested_struct() {
    println!("Testing empty nested struct...");

    let with_empty = WithEmpty {
        id: 42,
        empty: EmptyNested {},
        value: "test".to_string(),
    };

    let df = with_empty.to_dataframe().unwrap();

    // Should have columns for id and value, but empty struct should contribute 0 columns
    assert_eq!(df.shape(), (1, 2));

    let expected_columns = ["id", "value"];

    let column_names = df.get_column_names();
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in {:?}",
            expected_col,
            column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt32(42)
    );
    assert_eq!(
        df.column("value").unwrap().get(0).unwrap(),
        AnyValue::String("test")
    );
}

fn test_empty_dataframes() {
    println!("Testing empty dataframe functionality...");

    // Test empty dataframe for basic nested struct
    let person_empty = Person::empty_dataframe().unwrap();
    assert_eq!(person_empty.shape(), (0, 6));

    let expected_columns = [
        "name",
        "age",
        "address.street",
        "address.city",
        "address.zip",
        "salary",
    ];
    let column_names = person_empty.get_column_names();
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in empty dataframe",
            expected_col
        );
    }

    // Test empty dataframe for complex nested struct
    let employee_empty = Employee::empty_dataframe().unwrap();
    assert_eq!(employee_empty.shape(), (0, 10));

    // Test empty dataframe for deep nesting
    let company_empty = Company::empty_dataframe().unwrap();
    assert_eq!(company_empty.shape(), (0, 5));

    // Test empty dataframe for struct with vectors
    let developer_empty = Developer::empty_dataframe().unwrap();
    assert_eq!(developer_empty.shape(), (0, 4));

    // Test empty dataframe for struct with empty nested struct
    let with_empty_empty = WithEmpty::empty_dataframe().unwrap();
    assert_eq!(with_empty_empty.shape(), (0, 2));
}

fn test_data_types_and_values() {
    println!("Testing data types and values...");

    let employee = Employee {
        id: -12345, // Test negative i64
        name: "Test Employee".to_string(),
        address: Address {
            street: "123 Test St".to_string(),
            city: "Test City".to_string(),
            zip: 99999,
        },
        contact: Contact {
            email: None, // Test None values
            phone: None, // Test None values
            verified: false,
        },
        salary: 0.0, // Test zero value
        active: false,
    };

    let df = employee.to_dataframe().unwrap();

    // Test data types are preserved correctly
    assert_eq!(df.column("id").unwrap().dtype(), &DataType::Int64);
    assert_eq!(df.column("name").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("address.street").unwrap().dtype(),
        &DataType::String
    );
    assert_eq!(df.column("address.zip").unwrap().dtype(), &DataType::UInt32);
    assert_eq!(df.column("salary").unwrap().dtype(), &DataType::Float64);
    assert_eq!(df.column("active").unwrap().dtype(), &DataType::Boolean);
    assert_eq!(
        df.column("contact.verified").unwrap().dtype(),
        &DataType::Boolean
    );

    // Test specific values including edge cases
    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::Int64(-12345)
    );
    assert_eq!(
        df.column("salary").unwrap().get(0).unwrap(),
        AnyValue::Float64(0.0)
    );
    assert_eq!(
        df.column("active").unwrap().get(0).unwrap(),
        AnyValue::Boolean(false)
    );
    assert_eq!(
        df.column("contact.verified").unwrap().get(0).unwrap(),
        AnyValue::Boolean(false)
    );

    // Test None values are handled correctly
    let email_value = df.column("contact.email").unwrap().get(0).unwrap();
    let phone_value = df.column("contact.phone").unwrap().get(0).unwrap();
    assert!(matches!(email_value, AnyValue::Null));
    assert!(matches!(phone_value, AnyValue::Null));
}

fn test_multiple_nested_instances() {
    println!("Testing multiple instances of same nested type...");

    let multiple_addresses = MultipleAddresses {
        id: 123,
        home_address: Address {
            street: "123 Home St".to_string(),
            city: "Hometown".to_string(),
            zip: 11111,
        },
        work_address: Address {
            street: "456 Work Ave".to_string(),
            city: "Worktown".to_string(),
            zip: 22222,
        },
        name: "John Worker".to_string(),
    };

    let df = multiple_addresses.to_dataframe().unwrap();

    // Expected columns: id, home_address.*, work_address.*, name
    // id(1) + home_address(3) + work_address(3) + name(1) = 8 columns
    assert_eq!(df.shape(), (1, 8));

    let expected_columns = [
        "id",
        "home_address.street",
        "home_address.city",
        "home_address.zip",
        "work_address.street",
        "work_address.city",
        "work_address.zip",
        "name",
    ];

    let column_names = df.get_column_names();
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in {:?}",
            expected_col,
            column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    // Test that both instances have correct values
    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt32(123)
    );
    assert_eq!(
        df.column("name").unwrap().get(0).unwrap(),
        AnyValue::String("John Worker")
    );

    // Home address values
    assert_eq!(
        df.column("home_address.street").unwrap().get(0).unwrap(),
        AnyValue::String("123 Home St")
    );
    assert_eq!(
        df.column("home_address.city").unwrap().get(0).unwrap(),
        AnyValue::String("Hometown")
    );
    assert_eq!(
        df.column("home_address.zip").unwrap().get(0).unwrap(),
        AnyValue::UInt32(11111)
    );

    // Work address values
    assert_eq!(
        df.column("work_address.street").unwrap().get(0).unwrap(),
        AnyValue::String("456 Work Ave")
    );
    assert_eq!(
        df.column("work_address.city").unwrap().get(0).unwrap(),
        AnyValue::String("Worktown")
    );
    assert_eq!(
        df.column("work_address.zip").unwrap().get(0).unwrap(),
        AnyValue::UInt32(22222)
    );

    // Test empty dataframe
    let empty_df = MultipleAddresses::empty_dataframe().unwrap();
    assert_eq!(empty_df.shape(), (0, 8));
    assert_eq!(empty_df.get_column_names(), column_names);
}

fn test_complex_mixed_scenarios() {
    println!("Testing complex mixed scenarios...");

    let complex_mixed = ComplexMixed {
        id: 999,
        primary_contact: Some(Contact {
            email: Some("primary@example.com".to_string()),
            phone: None,
            verified: true,
        }),
        backup_contact: Contact {
            email: None,
            phone: Some("555-BACKUP".to_string()),
            verified: false,
        },
        addresses: vec!["123 Main St".to_string(), "456 Secondary St".to_string()],
        main_location: Location {
            name: "Main Office".to_string(),
            coordinates: Coordinates {
                latitude: 40.7128,
                longitude: -74.0060,
            },
        },
        active: true,
    };

    let df = complex_mixed.to_dataframe().unwrap();

    // Expected columns:
    // id(1) + primary_contact(3) + backup_contact(3) + addresses(1) + main_location(3) + active(1) = 12
    assert_eq!(df.shape(), (1, 12));

    let expected_columns = [
        "id",
        "primary_contact.email",
        "primary_contact.phone",
        "primary_contact.verified",
        "backup_contact.email",
        "backup_contact.phone",
        "backup_contact.verified",
        "addresses",
        "main_location.name",
        "main_location.coordinates.latitude",
        "main_location.coordinates.longitude",
        "active",
    ];

    let column_names = df.get_column_names();
    for expected_col in &expected_columns {
        assert!(
            column_names
                .iter()
                .any(|name| name.as_str() == *expected_col),
            "Column '{}' not found in {:?}",
            expected_col,
            column_names.iter().map(|s| s.as_str()).collect::<Vec<_>>()
        );
    }

    // Test specific values
    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::Int64(999)
    );
    assert_eq!(
        df.column("active").unwrap().get(0).unwrap(),
        AnyValue::Boolean(true)
    );

    // Test optional nested struct fields
    assert_eq!(
        df.column("primary_contact.email").unwrap().get(0).unwrap(),
        AnyValue::String("primary@example.com")
    );
    let primary_phone = df.column("primary_contact.phone").unwrap().get(0).unwrap();
    assert!(matches!(primary_phone, AnyValue::Null));
    assert_eq!(
        df.column("primary_contact.verified")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Boolean(true)
    );

    // Test non-optional nested struct fields
    let backup_email = df.column("backup_contact.email").unwrap().get(0).unwrap();
    assert!(matches!(backup_email, AnyValue::Null));
    assert_eq!(
        df.column("backup_contact.phone").unwrap().get(0).unwrap(),
        AnyValue::String("555-BACKUP")
    );
    assert_eq!(
        df.column("backup_contact.verified")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Boolean(false)
    );

    // Test deeply nested values
    assert_eq!(
        df.column("main_location.name").unwrap().get(0).unwrap(),
        AnyValue::String("Main Office")
    );
    assert_eq!(
        df.column("main_location.coordinates.latitude")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Float64(40.7128)
    );
    assert_eq!(
        df.column("main_location.coordinates.longitude")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Float64(-74.0060)
    );

    // Test vector field
    let addresses_value = df.column("addresses").unwrap().get(0).unwrap();
    assert!(matches!(addresses_value, AnyValue::List(_)));

    // Test empty dataframe
    let empty_df = ComplexMixed::empty_dataframe().unwrap();
    assert_eq!(empty_df.shape(), (0, 12));
    assert_eq!(empty_df.get_column_names(), column_names);
}
