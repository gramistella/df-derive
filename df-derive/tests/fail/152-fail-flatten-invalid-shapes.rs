use df_derive::ToDataFrame;

// `flatten` drops the field's own name, so the field must be a record: a
// type whose columns are named by its own fields.

#[derive(ToDataFrame)]
struct PrimitiveFlatten {
    #[df_derive(flatten)]
    primitive: i32,
}

#[derive(ToDataFrame)]
struct StringFlatten {
    #[df_derive(flatten)]
    label: String,
}

#[derive(ToDataFrame)]
struct OptionalPrimitiveFlatten {
    #[df_derive(flatten)]
    optional: Option<i32>,
}

fn main() {}
