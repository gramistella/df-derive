use df_derive::ToDataFrame;
use std::net::IpAddr;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct Unsupported {
    id: i32,
    // `IpAddr` has no column encoding of its own; it needs `as_string` or a
    // `Columnar` impl.
    address: IpAddr,
}

fn main() {}
