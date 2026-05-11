use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

mod time {
    pub struct Duration;
}

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(time_unit = "ns")]
    elapsed: time::Duration,
}

fn main() {}
