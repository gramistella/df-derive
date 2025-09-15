use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

#[derive(ToDataFrame, Clone, PartialEq, Debug)]
struct Item {
    id: u32,
    name: String,
}

#[derive(ToDataFrame)]
struct Container {
    id: i32,
    // A vec of optional primitives
    primitive_items: Vec<Option<i32>>,
    // A vec of optional custom structs
    custom_items: Vec<Option<Item>>,
    // An optional vec of optional primitives
    opt_vec_opt_primitive: Option<Vec<Option<i32>>>,
}

fn main() {
    println!("--- Testing complex wrapper combinations ---");

    let container = Container {
        id: 1,
        primitive_items: vec![Some(10), None, Some(30)],
        custom_items: vec![
            Some(Item { id: 100, name: "A".to_string() }),
            None,
            Some(Item { id: 300, name: "C".to_string() }),
        ],
        opt_vec_opt_primitive: Some(vec![Some(1), None, Some(3)]),
    };

    let df = container.to_dataframe().unwrap();
    println!("📊 DataFrame with complex wrappers:\n{}", df);

    // Expected columns: id, primitive_items, custom_items.id, custom_items.name, opt_vec_opt_primitive
    assert_eq!(df.shape(), (1, 5));

    // Verify schema
    let schema = df.schema();
    assert_eq!(schema.get("primitive_items").unwrap(), &DataType::List(Box::new(DataType::Int32)));
    assert_eq!(schema.get("custom_items.id").unwrap(), &DataType::List(Box::new(DataType::UInt32)));
    assert_eq!(schema.get("custom_items.name").unwrap(), &DataType::List(Box::new(DataType::String)));
    assert_eq!(schema.get("opt_vec_opt_primitive").unwrap(), &DataType::List(Box::new(DataType::Int32)));

    // Verify values for Vec<Option<i32>>
    let s_primitive = match df.column("primitive_items").unwrap().get(0).unwrap() {
        AnyValue::List(inner) => inner.clone(),
        _ => panic!("Expected List AnyValue for 'primitive_items'"),
    };
    let ca_primitive: &Int32Chunked = s_primitive.i32().unwrap();
    let vec_primitive: Vec<Option<i32>> = ca_primitive.into_iter().collect();
    assert_eq!(vec_primitive, vec![Some(10), None, Some(30)]);

    // Verify values for Vec<Option<Item>>
    let s_custom_id = match df.column("custom_items.id").unwrap().get(0).unwrap() {
        AnyValue::List(inner) => inner.clone(),
        _ => panic!("Expected List AnyValue for 'custom_items.id'"),
    };
    let ca_custom_id: &UInt32Chunked = s_custom_id.u32().unwrap();
    let vec_custom_id: Vec<Option<u32>> = ca_custom_id.into_iter().collect();
    assert_eq!(vec_custom_id, vec![Some(100), None, Some(300)]);
    
    let s_custom_name = match df.column("custom_items.name").unwrap().get(0).unwrap() {
        AnyValue::List(inner) => inner.clone(),
        _ => panic!("Expected List AnyValue for 'custom_items.name'"),
    };
    let ca_custom_name: &StringChunked = s_custom_name.str().unwrap();
    let vec_custom_name: Vec<Option<&str>> = ca_custom_name.into_iter().collect();
    assert_eq!(vec_custom_name, vec![Some("A"), None, Some("C")]);

    println!("✅ Complex wrapper combinations test passed!");
}