use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrame;

#[derive(ToDataFrame)]
struct Empty;

fn main() {
    println!("────────────────────────────────────────────────────────");
    println!("🔎 Testing empty struct → DataFrame...");

    let df = Empty.to_dataframe().unwrap();
    println!("\n📊 One-row DataFrame (0 columns) for empty struct:\n{}", df);
    assert_eq!(df.shape(), (1, 0));

    let empty_df = Empty::empty_dataframe().unwrap();
    println!("\n📄 Truly empty DataFrame: shape={:?}", empty_df.shape());
    assert_eq!(empty_df.shape(), (0, 0));

    println!("\n✅ Empty struct test completed successfully!");
}
