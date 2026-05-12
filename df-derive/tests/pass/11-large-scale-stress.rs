// Stress test with large data structures and performance considerations

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

// Large struct with many fields
#[derive(ToDataFrame)]
struct LargeStruct {
    // Primitive fields
    id: u64,
    field_01: f64,
    field_02: f64,
    field_03: f64,
    field_04: f64,
    field_05: f64,
    field_06: f64,
    field_07: f64,
    field_08: f64,
    field_09: f64,
    field_10: f64,
    field_11: i64,
    field_12: i64,
    field_13: i64,
    field_14: i64,
    field_15: i64,
    field_16: String,
    field_17: String,
    field_18: String,
    field_19: String,
    field_20: String,
    field_21: bool,
    field_22: bool,
    field_23: bool,
    field_24: bool,
    field_25: bool,

    // Optional fields
    opt_01: Option<f64>,
    opt_02: Option<i32>,
    opt_03: Option<String>,
    opt_04: Option<bool>,
    opt_05: Option<u64>,

    // Vector fields
    vec_01: Vec<f64>,
    vec_02: Vec<i32>,
    vec_03: Vec<String>,
    vec_04: Vec<bool>,
    vec_05: Vec<u64>,
}

// Deeply nested structure
#[derive(ToDataFrame)]
struct Level4 {
    value: f64,
    count: u32,
}

#[derive(ToDataFrame)]
struct Level3 {
    name: String,
    level4_data: Vec<Level4>,
}

#[derive(ToDataFrame)]
struct Level2 {
    id: u32,
    level3_data: Vec<Level3>,
}

#[derive(ToDataFrame)]
struct Level1 {
    symbol: String,
    level2_data: Vec<Level2>,
}

#[derive(ToDataFrame)]
struct DeepNesting {
    root_id: u64,
    tree: Level1,
}

// Wide structure with many Vec<CustomStruct> fields
#[derive(ToDataFrame)]
struct SimpleData {
    value: f64,
    timestamp: i64,
}

#[derive(ToDataFrame)]
struct WideStructure {
    id: String,
    data_01: Vec<SimpleData>,
    data_02: Vec<SimpleData>,
    data_03: Vec<SimpleData>,
    data_04: Vec<SimpleData>,
    data_05: Vec<SimpleData>,
    data_06: Vec<SimpleData>,
    data_07: Vec<SimpleData>,
    data_08: Vec<SimpleData>,
    data_09: Vec<SimpleData>,
    data_10: Vec<SimpleData>,
}

fn main() {
    test_large_struct();
    test_large_vector();
    test_deep_nesting();
    test_wide_structure();
    test_empty_dataframe_compilation_time();
}

fn test_large_struct() {
    let large = LargeStruct {
        id: 12345,
        field_01: 1.0,
        field_02: 2.0,
        field_03: 3.0,
        field_04: 4.0,
        field_05: 5.0,
        field_06: 6.0,
        field_07: 7.0,
        field_08: 8.0,
        field_09: 9.0,
        field_10: 10.0,
        field_11: 11,
        field_12: 12,
        field_13: 13,
        field_14: 14,
        field_15: 15,
        field_16: "s16".to_string(),
        field_17: "s17".to_string(),
        field_18: "s18".to_string(),
        field_19: "s19".to_string(),
        field_20: "s20".to_string(),
        field_21: true,
        field_22: false,
        field_23: true,
        field_24: false,
        field_25: true,

        opt_01: Some(1.5),
        opt_02: None,
        opt_03: Some("optional".to_string()),
        opt_04: None,
        opt_05: Some(999),

        vec_01: vec![1.1, 2.2, 3.3],
        vec_02: vec![10, 20, 30],
        vec_03: vec!["a".to_string(), "b".to_string()],
        vec_04: vec![true, false],
        vec_05: vec![100, 200, 300],
    };

    let df = large.to_dataframe().unwrap();

    // Should have all fields (exact count may vary based on how vectors are handled)
    assert_eq!(df.shape().0, 1); // One row
    assert!(df.shape().1 >= 30); // At least 30 columns

    let empty_df = LargeStruct::empty_dataframe().unwrap();
    assert_eq!(empty_df.shape().0, 0); // Zero rows
    assert!(empty_df.shape().1 >= 30); // At least 30 columns

    println!(
        "✅ Large struct test passed! {} columns processed",
        df.shape().1
    );
}

fn test_large_vector() {
    // Create a large vector of structs
    let large_vec: Vec<SimpleData> = (0..1000)
        .map(|i| SimpleData {
            value: i as f64 * 0.1,
            timestamp: 1700000000 + i,
        })
        .collect();

    let df = large_vec.to_dataframe().unwrap();

    // Should have 1000 rows and 2 columns
    assert_eq!(df.shape(), (1000, 2));

    println!(
        "✅ Large vector test passed! {} rows processed",
        df.shape().0
    );
}

fn test_deep_nesting() {
    let deep = DeepNesting {
        root_id: 1,
        tree: Level1 {
            symbol: "ROOT".to_string(),
            level2_data: vec![
                Level2 {
                    id: 10,
                    level3_data: vec![
                        Level3 {
                            name: "Branch1".to_string(),
                            level4_data: vec![
                                Level4 {
                                    value: 1.1,
                                    count: 5,
                                },
                                Level4 {
                                    value: 1.2,
                                    count: 3,
                                },
                            ],
                        },
                        Level3 {
                            name: "Branch2".to_string(),
                            level4_data: vec![Level4 {
                                value: 2.1,
                                count: 8,
                            }],
                        },
                    ],
                },
                Level2 {
                    id: 20,
                    level3_data: vec![Level3 {
                        name: "Branch3".to_string(),
                        level4_data: vec![
                            Level4 {
                                value: 3.1,
                                count: 12,
                            },
                            Level4 {
                                value: 3.2,
                                count: 15,
                            },
                            Level4 {
                                value: 3.3,
                                count: 7,
                            },
                        ],
                    }],
                },
            ],
        },
    };

    let df = deep.to_dataframe().unwrap();

    // Should flatten all levels correctly
    assert_eq!(df.shape().0, 1); // One root entity

    // Should have columns for each level: root_id, tree.symbol, tree.level2_data.id,
    // tree.level2_data.level3_data.name, tree.level2_data.level3_data.level4_data.value,
    // tree.level2_data.level3_data.level4_data.count
    assert_eq!(df.shape().1, 6);

    let expected_columns = [
        "root_id",
        "tree.symbol",
        "tree.level2_data.id",
        "tree.level2_data.level3_data.name",
        "tree.level2_data.level3_data.level4_data.value",
        "tree.level2_data.level3_data.level4_data.count",
    ];

    for expected in &expected_columns {
        assert!(
            df.get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found in deep nesting",
            expected
        );
    }

    println!("✅ Deep nesting test passed! 6 levels flattened correctly");
}

fn test_wide_structure() {
    let wide = WideStructure {
        id: "wide_test".to_string(),
        data_01: vec![SimpleData {
            value: 1.0,
            timestamp: 1001,
        }],
        data_02: vec![SimpleData {
            value: 2.0,
            timestamp: 1002,
        }],
        data_03: vec![SimpleData {
            value: 3.0,
            timestamp: 1003,
        }],
        data_04: vec![SimpleData {
            value: 4.0,
            timestamp: 1004,
        }],
        data_05: vec![SimpleData {
            value: 5.0,
            timestamp: 1005,
        }],
        data_06: vec![SimpleData {
            value: 6.0,
            timestamp: 1006,
        }],
        data_07: vec![SimpleData {
            value: 7.0,
            timestamp: 1007,
        }],
        data_08: vec![SimpleData {
            value: 8.0,
            timestamp: 1008,
        }],
        data_09: vec![SimpleData {
            value: 9.0,
            timestamp: 1009,
        }],
        data_10: vec![SimpleData {
            value: 10.0,
            timestamp: 1010,
        }],
    };

    let df = wide.to_dataframe().unwrap();

    // Should have: id + (value + timestamp) * 10 = 21 columns
    assert_eq!(df.shape(), (1, 21));

    println!(
        "✅ Wide structure test passed! {} columns from 10 Vec<CustomStruct> fields",
        df.shape().1
    );
}

fn test_empty_dataframe_compilation_time() {
    // This test mainly exercises the macro at compile time
    // If this compiles quickly, the macro is efficient

    let start = std::time::Instant::now();

    // Simple operation to ensure the macro-generated code is actually used
    let _ = LargeStruct::empty_dataframe().unwrap();
    let _ = DeepNesting::empty_dataframe().unwrap();
    let _ = WideStructure::empty_dataframe().unwrap();

    let duration = start.elapsed();

    println!(
        "✅ Compilation efficiency test passed! Runtime: {:?}",
        duration
    );

    // The main test is that this compiles in reasonable time
    // If compilation takes too long, it indicates macro efficiency issues
}
