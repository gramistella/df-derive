use df_derive::ToDataFrame;


#[derive(ToDataFrame)]
struct Bad<'a> {
    #[df_derive(as_binary)]
    items: &'a [i32],
}

fn main() {}
