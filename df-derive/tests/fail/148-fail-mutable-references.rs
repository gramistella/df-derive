use df_derive::ToDataFrame;

#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct DirectMutRef<'a> {
    field: &'a mut i32,
}

#[derive(ToDataFrame)]
struct OptionMutRef<'a> {
    field: Option<&'a mut String>,
}

#[derive(ToDataFrame)]
struct VecMutRef<'a> {
    field: Vec<&'a mut u32>,
}

#[derive(ToDataFrame)]
struct MutStr<'a> {
    field: &'a mut str,
}

#[derive(ToDataFrame)]
struct MutBytes<'a> {
    #[df_derive(as_binary)]
    field: &'a mut [u8],
}

fn main() {}
