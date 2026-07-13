use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
union Bits {
    a: u32,
    b: f32,
}

fn main() {}
