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
struct CrateQualified {
    id: u64,
    child: Option<Box<crate::CrateQualified>>,
}

#[derive(ToDataFrame)]
struct SelfQualified {
    id: u64,
    child: Option<Box<self::SelfQualified>>,
}

#[derive(ToDataFrame)]
struct TupleField {
    nested: (u64, Option<Box<TupleField>>),
}

#[derive(ToDataFrame)]
struct TupleStruct(u64, Option<Box<TupleStruct>>);

#[derive(ToDataFrame)]
struct GenericNode<T> {
    id: u64,
    #[df_derive(skip)]
    marker: std::marker::PhantomData<T>,
    child: Option<Box<GenericNode<T>>>,
}

#[derive(ToDataFrame)]
struct CrateQualifiedGeneric<T> {
    id: u64,
    #[df_derive(skip)]
    marker: std::marker::PhantomData<T>,
    child: Option<Box<crate::CrateQualifiedGeneric<T>>>,
}

#[derive(ToDataFrame)]
struct SelfQualifiedGeneric<T> {
    id: u64,
    #[df_derive(skip)]
    marker: std::marker::PhantomData<T>,
    child: Option<Box<self::SelfQualifiedGeneric<T>>>,
}

fn main() {}
