use syn::{Ident, Type};

use super::{LeafSpec, WrapperShape};

/// Top-level IR for a struct targeted by the derive.
#[derive(Clone)]
pub struct StructIR {
    /// The identifier of the struct being derived
    pub name: Ident,
    /// The generics declared on the struct (empty when no generics are used)
    pub generics: syn::Generics,
    /// The fields of the struct in declaration order
    pub fields: Vec<FieldIR>,
}

/// IR for a single field of a struct
#[derive(Clone)]
pub struct FieldIR {
    /// Field name as declared on the struct
    pub name: Ident,
    /// Field index for tuple structs (None for named fields)
    pub field_index: Option<usize>,
    /// Per-leaf semantic shape.
    pub leaf_spec: LeafSpec,
    /// Per-wrapper semantic shape.
    pub wrapper_shape: WrapperShape,
    /// Generic type parameters that were explicitly opted into decimal
    /// encoding with `#[df_derive(decimal(...))]`.
    pub decimal_generic_params: Vec<Ident>,
    /// Concrete custom backend type explicitly opted into decimal encoding.
    pub decimal_backend_ty: Option<Type>,
    /// Number of transparent pointer layers peeled at the outer position.
    pub outer_smart_ptr_depth: usize,
}
