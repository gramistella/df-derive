use super::{LeafSpec, WrapperShape};

/// One element of a tuple-typed field. Each element contributes one or more
/// columns to the parent struct's flattened layout.
#[derive(Clone)]
pub struct TupleElement {
    /// Element's leaf classification.
    pub leaf_spec: LeafSpec,
    /// Element's wrapper shape.
    pub wrapper_shape: WrapperShape,
    /// Smart-pointer depth above any wrapper in the element type.
    pub outer_smart_ptr_depth: usize,
}
