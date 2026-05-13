use df_derive::ToDataFrame;
use std::collections::{BTreeSet, LinkedList, VecDeque};
#[path = "../common.rs"]
mod core;

#[derive(ToDataFrame)]
struct UnsupportedBTreeSet {
    values: BTreeSet<String>,
}

#[derive(ToDataFrame)]
struct UnsupportedVecDeque {
    values: VecDeque<String>,
}

#[derive(ToDataFrame)]
struct UnsupportedLinkedList {
    values: LinkedList<String>,
}

fn main() {}
