use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct WithAsStringOnTuple {
    #[df_derive(as_string)]
    pair: (String, i32),
}

fn main() {}
