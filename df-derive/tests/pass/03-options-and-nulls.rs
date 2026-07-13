use df_derive::ToDataFrame;
use polars::prelude::*;
use df_derive::dataframe::ToDataFrame;

#[derive(ToDataFrame)]
struct Nullable {
    name: String,
    value: Option<i32>,
}

fn main() {
    println!("────────────────────────────────────────────────────────");
    println!("🔎 Testing options and nulls handling...");

    let some_instance = Nullable {
        name: "A".to_string(),
        value: Some(100),
    };
    let none_instance = Nullable {
        name: "B".to_string(),
        value: None,
    };

    let df_some = some_instance.to_dataframe().unwrap();
    println!("\n📊 DataFrame (Some):\n{}", df_some);
    assert_eq!(
        df_some.column("value").unwrap().get(0).unwrap(),
        AnyValue::Int32(100)
    );

    let df_none = none_instance.to_dataframe().unwrap();
    println!("\n📊 DataFrame (None):\n{}", df_none);
    assert!(df_none.column("value").unwrap().get(0).unwrap().is_null());

    println!("\n✅ Options and nulls test completed successfully!");
}
