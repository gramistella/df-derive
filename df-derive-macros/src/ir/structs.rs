use syn::Ident;

use super::{ColumnIR, LeafSpec, WrapperShape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructIR {
    pub name: Ident,
    pub generics: syn::Generics,
    pub columns: Vec<ColumnIR>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIR {
    pub name: Ident,
    pub field_index: Option<usize>,
    pub leaf_spec: LeafSpec,
    pub wrapper_shape: WrapperShape,
    pub outer_smart_ptr_depth: usize,
}
