use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad<'a> {
    #[df_derive(as_string)]
    items: &'a [u16],
}

fn main() {}
