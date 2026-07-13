use df_derive::ToDataFrame;

use chrono::{DateTime, Utc};

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(time_unit = "seconds")]
    ts: DateTime<Utc>,
}

fn main() {}
