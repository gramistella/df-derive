use df_derive::ToDataFrame;

// An enum named by its own text has to have that text: the error is at the
// attribute that asks for it.
#[derive(ToDataFrame)]
#[df_derive(as_str)]
enum Unborrowed {
    One,
}

#[derive(ToDataFrame)]
#[df_derive(as_string)]
enum Unformatted {
    One,
}

fn main() {}
