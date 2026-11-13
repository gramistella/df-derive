use df_derive::ToDataFrame;

mod other {
    /// Not a column type, and named like the parameter of `Holder`.
    pub struct Data;
}

// `other::Data` is an item, not the parameter `Data`. The field does not
// depend on the parameter, so it is checked here and not where the type is
// first used.
#[derive(ToDataFrame)]
struct Holder<Data> {
    item: other::Data,
    parameter: Data,
}

fn main() {}
