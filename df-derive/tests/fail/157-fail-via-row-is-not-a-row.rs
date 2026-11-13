use df_derive::ToDataFrame;

// A projected type is a row type, so the type it encodes as must be one: a
// struct or tuple of columns. A single value has no column names of its own.
// `transparent` is how a wrapper encodes as its one field.

#[derive(ToDataFrame)]
#[df_derive(via = "i32")]
struct Meters(i32);

impl From<&Meters> for i32 {
    fn from(meters: &Meters) -> Self {
        meters.0
    }
}

fn main() {}
