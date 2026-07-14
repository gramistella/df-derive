use syn::Ident;

use super::{FieldPlan, LeafSpec, NestedNamePolicy, TerminalColumnRef, WrapperShape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructIR {
    pub name: Ident,
    pub generics: syn::Generics,
    pub fields: Vec<FieldPlan>,
}

impl StructIR {
    pub fn visit_terminal_columns(&self, mut visitor: impl FnMut(TerminalColumnRef<'_>)) {
        for field in &self.fields {
            field.visit_terminal_columns(&mut visitor);
        }
    }

    pub fn terminal_column_count(&self) -> usize {
        self.fields
            .iter()
            .map(FieldPlan::terminal_column_count)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIR {
    pub name: Ident,
    pub field_index: Option<usize>,
    pub leaf_spec: LeafSpec,
    pub wrapper_shape: WrapperShape,
    pub outer_smart_ptr_depth: usize,
    pub nested_name_policy: NestedNamePolicy,
}
