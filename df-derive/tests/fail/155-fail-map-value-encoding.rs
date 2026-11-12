// An encoding attribute on a map applies to the map's values. Its keys keep
// the encoding of their own type.
use df_derive::ToDataFrame;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Side {
    Buy,
    Sell,
}

impl AsRef<str> for Side {
    fn as_ref(&self) -> &str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
        }
    }
}

#[derive(ToDataFrame)]
struct Book {
    // `as_str` is tried on the numbers, where it does not apply.
    #[df_derive(as_str)]
    depth: BTreeMap<String, f64>,
    // The values are fine here, but `Side` is a key and has no encoding of
    // its own: the attribute does not reach it.
    #[df_derive(as_string)]
    levels: BTreeMap<Side, f64>,
}

fn main() {}
