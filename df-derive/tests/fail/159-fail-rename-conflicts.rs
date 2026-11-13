use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
struct Inner {
    value: i32,
}

// A skipped field has no column to name.
#[derive(ToDataFrame)]
struct RenameSkip {
    #[df_derive(rename = "kept", skip)]
    skipped: i32,
}

// A flattened field adds no name of its own, and a prefix already is one.
#[derive(ToDataFrame)]
struct RenameFlatten {
    #[df_derive(rename = "inner", flatten)]
    flat: Inner,
}

#[derive(ToDataFrame)]
struct RenameFlattenPrefix {
    #[df_derive(flatten(prefix = "a"), rename = "b")]
    flat: Inner,
}

#[derive(ToDataFrame)]
struct EmptyRename {
    #[df_derive(rename = "")]
    value: i32,
}

#[derive(ToDataFrame)]
struct DuplicateRename {
    #[df_derive(rename = "a", rename = "b")]
    value: i32,
}

#[derive(ToDataFrame)]
struct RenameNotAString {
    #[df_derive(rename = 5)]
    value: i32,
}

// The one field of a transparent wrapper takes the holder's name.
#[derive(ToDataFrame)]
#[df_derive(transparent)]
struct RenameInTransparent(#[df_derive(rename = "inner")] i32);

fn main() {}
