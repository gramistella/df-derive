use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(time_unit = "ns")]
    not_a_datetime: i64,
}

fn main() {}
