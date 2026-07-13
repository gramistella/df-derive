use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct WithUnit {
    nothing: (),
}

fn main() {}
