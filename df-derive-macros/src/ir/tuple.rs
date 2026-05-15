use super::{LeafSpec, WrapperShape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleElement {
    pub leaf_spec: LeafSpec,
    pub wrapper_shape: WrapperShape,
    pub outer_smart_ptr_depth: usize,
}
