use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_string)]
    duration: std::time::Duration,
}

fn main() {}
