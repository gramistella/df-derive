use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad<'a> {
    #[df_derive(as_string)]
    bytes: &'a [u8],
}

fn main() {}
