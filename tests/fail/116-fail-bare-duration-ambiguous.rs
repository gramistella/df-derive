use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

// Bring `Duration` into scope to mimic the realistic ambiguous case where
// both crates' Durations are reachable. We never construct a `Duration`
// value here — the macro must reject the field at parse time before any
// runtime code is generated, so `Duration` only needs to resolve as a
// type-name to make the file's Rust syntax valid.
#[allow(unused_imports)]
use chrono::Duration;

#[derive(ToDataFrame)]
struct Bad {
    elapsed: Duration,
}

fn main() {}
