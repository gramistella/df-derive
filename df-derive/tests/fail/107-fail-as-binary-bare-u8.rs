use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary)]
    one_byte: u8,
}

fn main() {}
