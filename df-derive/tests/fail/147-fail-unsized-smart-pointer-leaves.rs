use std::rc::Rc;
use std::sync::Arc;

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct BoxedStr {
    label: Box<str>,
}

#[derive(ToDataFrame)]
struct ArcStr {
    label: Arc<str>,
}

#[derive(ToDataFrame)]
struct RcSlice {
    values: Rc<[i32]>,
}

#[derive(ToDataFrame)]
struct BoxedBytes {
    #[df_derive(as_binary)]
    blob: Option<Box<[u8]>>,
}

fn main() {}
