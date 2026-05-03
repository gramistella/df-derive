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
    // A vec of vecs of primitives — exercises the encoder IR's stacked
    // `LargeListArray` path for two-deep `Vec` shapes.
    nested_primitive: Vec<Vec<i32>>,
    // A vec of vecs of vecs — locks the recursive `ListBuilder` nesting.
    triple_nested: Vec<Vec<Vec<i32>>>,
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
        nested_primitive: vec![vec![1, 2, 3], vec![], vec![10, 20]],
        triple_nested: vec![vec![vec![1, 2], vec![3]], vec![vec![]], vec![]],
    };

    let df = container.to_dataframe().unwrap();
    println!("📊 DataFrame with complex wrappers:\n{}", df);

    // Expected columns: id, primitive_items, custom_items.id, custom_items.name, opt_vec_opt_primitive, nested_primitive, triple_nested
    assert_eq!(df.shape(), (1, 7));

    // Verify schema
    let schema = df.schema();
    assert_eq!(schema.get("primitive_items").unwrap(), &DataType::List(Box::new(DataType::Int32)));
    assert_eq!(schema.get("custom_items.id").unwrap(), &DataType::List(Box::new(DataType::UInt32)));
    assert_eq!(schema.get("custom_items.name").unwrap(), &DataType::List(Box::new(DataType::String)));
    assert_eq!(schema.get("opt_vec_opt_primitive").unwrap(), &DataType::List(Box::new(DataType::Int32)));
    // The runtime DataFrame's schema reflects the actual nesting depth (one
    // List<> per `Vec<>` layer in the field type). The macro's static
    // `schema()` collapses to a single List<element> regardless of depth (a
    // known limitation), but `df.schema()` here reads the column dtypes that
    // the typed `ListBuilder` chain emits, which carry the full nesting.
    assert_eq!(
        schema.get("nested_primitive").unwrap(),
        &DataType::List(Box::new(DataType::List(Box::new(DataType::Int32))))
    );
    assert_eq!(
        schema.get("triple_nested").unwrap(),
        &DataType::List(Box::new(DataType::List(Box::new(DataType::List(Box::new(
            DataType::Int32
        ))))))
    );

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

    // Verify values for Vec<Vec<i32>> — drill through the outer list to confirm
    // each inner Series is a typed Int32, not a fallback dtype-Null Series. A
    // regression that loses the typed `ListBuilder` path would produce one of:
    //   - `List<Null>` (empty inner, fallback empty path)
    //   - `Series` of `AnyValue` re-inferred to `List<List<Int32>>` mismatch
    let s_nested = match df.column("nested_primitive").unwrap().get(0).unwrap() {
        AnyValue::List(inner) => inner.clone(),
        _ => panic!("Expected List AnyValue for 'nested_primitive'"),
    };
    assert_eq!(s_nested.dtype(), &DataType::List(Box::new(DataType::Int32)));
    assert_eq!(s_nested.len(), 3);
    let nested_rows: Vec<Vec<Option<i32>>> = (0..s_nested.len())
        .map(|i| match s_nested.get(i).unwrap() {
            AnyValue::List(inner) => inner.i32().unwrap().into_iter().collect(),
            _ => panic!("Expected List AnyValue for inner row"),
        })
        .collect();
    assert_eq!(
        nested_rows,
        vec![
            vec![Some(1), Some(2), Some(3)],
            vec![],
            vec![Some(10), Some(20)],
        ],
    );

    // Vec<Vec<Vec<i32>>>: outermost cell → list of list of i32 → list of i32.
    let s_triple = match df.column("triple_nested").unwrap().get(0).unwrap() {
        AnyValue::List(inner) => inner.clone(),
        _ => panic!("Expected List AnyValue for 'triple_nested'"),
    };
    assert_eq!(
        s_triple.dtype(),
        &DataType::List(Box::new(DataType::List(Box::new(DataType::Int32))))
    );
    assert_eq!(s_triple.len(), 3);

    println!("✅ Complex wrapper combinations test passed!");
}