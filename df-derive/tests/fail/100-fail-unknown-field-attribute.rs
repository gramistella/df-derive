use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Typo {
    #[df_derive(as_strg)]
    s: String,
}

fn main() {}
