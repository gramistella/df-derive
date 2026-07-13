use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary, as_string)]
    bytes: Vec<u8>,
}

fn main() {}
