use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

// A tuple struct that will be nested
#[derive(ToDataFrame)]
struct Coords(f64, f64);

// A named struct that will be nested
#[derive(ToDataFrame)]
struct Attributes {
    is_active: bool,
    tag: String,
}

// A named struct containing a nested tuple struct
#[derive(ToDataFrame)]
struct Place {
    name: String,
    coords: Coords,
}

// A tuple struct containing a nested named struct
#[derive(ToDataFrame)]
struct Item(u32, String, Attributes);

// A complex struct mixing both
#[derive(ToDataFrame)]
struct Asset {
    id: i64,
    place: Place,
    item_data: Item,
}

fn main() {
    println!("--- Testing nesting with tuple structs ---");

    // Test 1: Named struct containing a tuple struct
    let place = Place {
        name: "Eiffel Tower".to_string(),
        coords: Coords(48.8584, 2.2945),
    };
    let df_place = place.to_dataframe().unwrap();
    println!("📊 Place DataFrame:\n{}", df_place);
    assert_eq!(df_place.shape(), (1, 3));
    let expected_place_cols = ["name", "coords.field_0", "coords.field_1"];
    assert_eq!(df_place.get_column_names(), expected_place_cols);
    assert_eq!(df_place.column("coords.field_0").unwrap().get(0).unwrap(), AnyValue::Float64(48.8584));

    // Test 2: Tuple struct containing a named struct
    let item = Item(
        101, 
        "Widget".to_string(), 
        Attributes { is_active: true, tag: "A".to_string() }
    );
    let df_item = item.to_dataframe().unwrap();
    println!("📊 Item DataFrame:\n{}", df_item);
    assert_eq!(df_item.shape(), (1, 4));
    let expected_item_cols = ["field_0", "field_1", "field_2.is_active", "field_2.tag"];
    assert_eq!(df_item.get_column_names(), expected_item_cols);
    assert_eq!(df_item.column("field_2.is_active").unwrap().get(0).unwrap(), AnyValue::Boolean(true));

    // Test 3: Complex combination
    let asset = Asset {
        id: 999,
        place,
        item_data: item,
    };
    let df_asset = asset.to_dataframe().unwrap();
    println!("📊 Asset DataFrame:\n{}", df_asset);
    assert_eq!(df_asset.shape(), (1, 8));
    let expected_asset_cols = [
        "id",
        "place.name",
        "place.coords.field_0",
        "place.coords.field_1",
        "item_data.field_0",
        "item_data.field_1",
        "item_data.field_2.is_active",
        "item_data.field_2.tag",
    ];
    assert_eq!(df_asset.get_column_names(), expected_asset_cols);
    assert_eq!(df_asset.column("id").unwrap().get(0).unwrap(), AnyValue::Int64(999));
    assert_eq!(df_asset.column("place.coords.field_1").unwrap().get(0).unwrap(), AnyValue::Float64(2.2945));
    assert_eq!(df_asset.column("item_data.field_0").unwrap().get(0).unwrap(), AnyValue::UInt32(101));
    assert_eq!(df_asset.column("item_data.field_2.tag").unwrap().get(0).unwrap(), AnyValue::String("A"));
    
    println!("\n✅ Nesting with tuple structs test passed!");
}