use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_binary)]
    ints: Vec<i32>,
}

fn main() {}
