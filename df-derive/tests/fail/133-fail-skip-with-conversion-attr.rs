use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct BadSkip {
    #[df_derive(skip, as_str)]
    value: String,
}

fn main() {}
