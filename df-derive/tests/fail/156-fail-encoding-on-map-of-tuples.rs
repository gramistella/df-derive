// The values of a map are where an encoding attribute applies, and a tuple
// cannot carry one there either.
use df_derive::ToDataFrame;
use std::collections::BTreeMap;

#[derive(ToDataFrame)]
struct WithAsStringOnMapOfTuples {
    #[df_derive(as_string)]
    ranges: BTreeMap<String, (f64, f64)>,
}

fn main() {}
