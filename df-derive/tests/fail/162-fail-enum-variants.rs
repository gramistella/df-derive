use df_derive::ToDataFrame;

// A variant takes one attribute, its name.
#[derive(ToDataFrame)]
enum Skipped {
    Kept,
    #[df_derive(skip)]
    Left,
}

#[derive(ToDataFrame)]
enum Text {
    #[df_derive(as_str)]
    Plain,
}

#[derive(ToDataFrame)]
enum Twice {
    #[df_derive(rename = "one", rename = "uno")]
    One,
}

#[derive(ToDataFrame)]
enum Unnamed {
    #[df_derive(rename)]
    One,
}

#[derive(ToDataFrame)]
enum Numbered {
    #[df_derive(rename = 1)]
    One,
}

// A name stands for one variant, whether it is the variant's own or given.
#[derive(ToDataFrame)]
enum Taken {
    Buy,
    #[df_derive(rename = "Buy")]
    Bid,
}

#[derive(ToDataFrame)]
enum Shared {
    #[df_derive(rename = "x")]
    Left,
    #[df_derive(rename = "x")]
    Right,
}

// An enum is one column already.
#[derive(ToDataFrame)]
#[df_derive(transparent)]
enum Wrapped {
    Only,
}

// An enum that takes its names from its own text takes all of them there,
// and from one place.
#[derive(ToDataFrame)]
#[df_derive(as_str)]
enum Mixed {
    #[df_derive(rename = "one")]
    One,
    Two,
}

impl AsRef<str> for Mixed {
    fn as_ref(&self) -> &str {
        "mixed"
    }
}

#[derive(ToDataFrame)]
#[df_derive(as_str, as_string)]
enum Both {
    One,
}

// On a struct there are no variants to name.
#[derive(ToDataFrame)]
#[df_derive(as_string)]
struct Row {
    id: u32,
}

fn main() {}
