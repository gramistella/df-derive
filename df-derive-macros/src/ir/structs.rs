use syn::{Ident, Type};

use super::{LeafSpec, WrapperShape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructIR {
    pub name: Ident,
    pub generics: syn::Generics,
    pub fields: Vec<FieldIR>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIR {
    pub name: Ident,
    pub field_index: Option<usize>,
    pub leaf_spec: LeafSpec,
    pub wrapper_shape: WrapperShape,
    /// Generic type parameters that were explicitly opted into decimal
    /// encoding with `#[df_derive(decimal(...))]`.
    pub decimal_generic_params: Vec<Ident>,
    pub decimal_backend_ty: Option<Type>,
    pub outer_smart_ptr_depth: usize,
}
