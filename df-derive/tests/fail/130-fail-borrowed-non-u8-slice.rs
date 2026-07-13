use df_derive::ToDataFrame;


#[derive(ToDataFrame)]
struct Bad<'a> {
    items: &'a [i32],
}

fn main() {}
