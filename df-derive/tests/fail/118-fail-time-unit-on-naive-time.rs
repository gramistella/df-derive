use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

use chrono::NaiveTime;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(time_unit = "us")]
    at: NaiveTime,
}

fn main() {}
