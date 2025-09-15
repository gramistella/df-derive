use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

#[derive(ToDataFrame)]
struct Simple {
    price: f64,
    volume: u64,
    name: String,
}

fn main() {
    println!("────────────────────────────────────────────────────────");
    println!("🔎 Testing simple struct → DataFrame...");

    let s = Simple {
        price: 150.75,
        volume: 1_000_000,
        name: "Widget".to_string(),
    };

    let df = s.to_dataframe().unwrap();
    println!("\n📊 Resulting DataFrame:\n{}", df);
    assert_eq!(df.shape(), (1, 3));

    let empty_df = Simple::empty_dataframe().unwrap();
    println!("\n📄 Empty DataFrame schema columns: {:?}", empty_df.get_column_names());
    assert_eq!(empty_df.shape(), (0, 3));
    assert_eq!(empty_df.get_column_names(), &["price", "volume", "name"]);

    println!("\n✅ Simple struct test completed successfully!");
}
