use df_derive::ToDataFrame;

// An enum is one column when its variants hold no data. One that holds some
// is not, and is told the two ways there are to store it.
#[derive(ToDataFrame)]
enum Status {
    Active,
    Suspended { reason: String },
}

#[derive(ToDataFrame)]
enum Reading {
    Missing,
    Value(f64),
}

fn main() {}
