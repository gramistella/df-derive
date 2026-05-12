use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct WithTimeUnitOnTuple {
    #[df_derive(time_unit = "us")]
    pair: (chrono::DateTime<chrono::Utc>, String),
}

fn main() {}
