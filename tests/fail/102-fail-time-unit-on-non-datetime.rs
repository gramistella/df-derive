use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(time_unit = "ns")]
    not_a_datetime: i64,
}

fn main() {}
