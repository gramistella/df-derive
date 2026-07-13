use df_derive::ToDataFrame;


#[derive(ToDataFrame)]
struct Bad<'a> {
    bs: &'a [u8],
}

fn main() {}
