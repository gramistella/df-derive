use syn::Ident;

use super::{AccessChain, LeafSpec, TerminalLeafSpec, WrapperShape};

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

/// Codegen-ready terminal column.
///
/// Invariants:
/// - `leaf_spec` is never `LeafSpec::Tuple`.
/// - Tuple fields are flattened into one `ColumnIR` per terminal element.
/// - `ParentVec` projections have a single terminal tuple projection step.
/// - `ParentOption` projections compose the parent option into `wrapper_shape`.
/// - `wrapper_shape.vec_depth()` already includes parent tuple-list layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnIR {
    pub name: String,
    pub source: ColumnSource,
    pub leaf_spec: TerminalLeafSpec,
    pub wrapper_shape: WrapperShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnSource {
    Field(FieldSource),
    TupleProjection {
        root: FieldSource,
        path: Vec<TupleProjectionStep>,
        context: ProjectionContext,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldSource {
    pub name: Ident,
    pub field_index: Option<usize>,
    pub outer_smart_ptr_depth: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleProjectionStep {
    pub index: usize,
    pub outer_smart_ptr_depth: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectionContext {
    Static,
    ParentOption {
        access: AccessChain,
    },
    ParentVec {
        projection_layer: usize,
        parent_inner_access: AccessChain,
    },
}
