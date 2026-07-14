use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrame as _;

#[derive(ToDataFrame)]
struct WrappedParent {
    nested: Option<((i32, String), bool)>,
}

#[derive(ToDataFrame)]
struct WrappedElement {
    nested: (Option<(i32, String)>, bool),
}

#[derive(ToDataFrame)]
struct WrappedVec {
    nested: Vec<((i32, String), bool)>,
}

fn main() {
    WrappedParent {
        nested: Some(((1, "parent".to_owned()), true)),
    }
    .to_dataframe()
    .unwrap();

    WrappedElement {
        nested: (Some((2, "element".to_owned())), false),
    }
    .to_dataframe()
    .unwrap();

    WrappedVec {
        nested: vec![((3, "vec".to_owned()), true)],
    }
    .to_dataframe()
    .unwrap();
}
