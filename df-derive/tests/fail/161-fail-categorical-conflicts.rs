use df_derive::ToDataFrame;

// A categorical column holds text. It goes with `as_str` and `as_string`,
// the two ways of reading text, and with no other encoding.
#[derive(ToDataFrame)]
struct WithBinary {
    #[df_derive(as_binary, categorical)]
    payload: Vec<u8>,
}

#[derive(ToDataFrame)]
struct WithDecimal {
    #[df_derive(categorical, decimal(precision = 10, scale = 2))]
    amount: String,
}

#[derive(ToDataFrame)]
struct WithTimeUnit {
    #[df_derive(time_unit = "ms", categorical)]
    elapsed: std::time::Duration,
}

#[derive(ToDataFrame)]
struct Twice {
    #[df_derive(categorical, categorical)]
    venue: String,
}

#[derive(ToDataFrame)]
struct Skipped {
    #[df_derive(skip, categorical)]
    venue: String,
}

#[derive(ToDataFrame)]
struct Inner {
    venue: String,
}

// It applies to a value, not to the columns of a nested row.
#[derive(ToDataFrame)]
struct Flattened {
    #[df_derive(flatten, categorical)]
    inner: Inner,
}

// The value has to be text.
#[derive(ToDataFrame)]
struct NotText {
    #[df_derive(categorical)]
    count: u32,
}

struct Opaque;

#[derive(ToDataFrame)]
struct NotShown {
    #[df_derive(as_string, categorical)]
    value: Opaque,
}

fn main() {}
