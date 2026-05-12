use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

use chrono::NaiveDate;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(time_unit = "ms")]
    when: NaiveDate,
}

fn main() {}
