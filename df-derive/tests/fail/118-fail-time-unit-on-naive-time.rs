use df_derive::ToDataFrame;

use chrono::NaiveTime;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(time_unit = "us")]
    at: NaiveTime,
}

fn main() {}
