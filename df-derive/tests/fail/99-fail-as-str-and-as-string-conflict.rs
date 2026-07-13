use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Conflict {
    #[df_derive(as_str)]
    #[df_derive(as_string)]
    name: String,
}

fn main() {}
