use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary)]
    s: String,
}

fn main() {}
