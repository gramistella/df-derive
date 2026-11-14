use df_derive::ToDataFrame;

// The type of an `Enum` column is the list of the enum's variants. An enum
// that says it may be given another has to say how it is stored instead.
#[derive(ToDataFrame)]
#[non_exhaustive]
enum Growing {
    One,
    Two,
}

/// The attribute is found among others, and whatever the names come from.
#[derive(ToDataFrame, Clone, Copy)]
#[df_derive(as_string)]
#[non_exhaustive]
enum Named {
    One,
}

// `categorical` on a type is an enum's. A field of a struct takes its own.
#[derive(ToDataFrame)]
#[df_derive(categorical)]
struct Row {
    name: String,
}

#[derive(ToDataFrame)]
#[df_derive(categorical, categorical)]
enum Twice {
    One,
}

// A projected type encodes as its row type, which says how it is stored.
#[derive(ToDataFrame)]
#[df_derive(categorical, via = "Flat")]
enum Projected {
    One,
}

#[derive(ToDataFrame)]
struct Flat {
    one: bool,
}

impl From<&Projected> for Flat {
    fn from(_: &Projected) -> Self {
        Self { one: true }
    }
}

// What is stored is the name of a variant, and a variant that holds data
// is more than a name.
#[derive(ToDataFrame)]
#[df_derive(categorical)]
enum Loaded {
    Empty,
    Full(u8),
}

// What an enum can say about itself is listed where it says something else.
#[derive(ToDataFrame)]
#[df_derive(ordered)]
enum Unknown {
    One,
}

fn main() {}
