use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
enum Status {
    Active,
    Inactive,
}

fn main() {}
