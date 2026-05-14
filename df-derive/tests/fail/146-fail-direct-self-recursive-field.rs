use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Node {
    id: u64,
    child: Option<Box<Node>>,
}

#[derive(ToDataFrame)]
struct SelfNamed {
    id: u64,
    child: Option<Box<Self>>,
}

#[derive(ToDataFrame)]
struct TupleField {
    nested: (u64, Option<Box<TupleField>>),
}

#[derive(ToDataFrame)]
struct TupleStruct(u64, Option<Box<TupleStruct>>);

fn main() {}
