use syn::Ident;

use super::{NestedNamePolicy, NonEmpty, TerminalLeafSpec, WrapperShape};

/// One source field in declaration order.
///
/// Tuple fields retain their semantic hierarchy until execution planning so
/// sibling leaves can share source traversal and list infrastructure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldPlan {
    Column(FieldColumn),
    Tuple(TupleField),
}

impl FieldPlan {
    pub(crate) const fn column(
        name: String,
        source: FieldSource,
        leaf_spec: TerminalLeafSpec,
        wrapper_shape: WrapperShape,
        nested_name_policy: NestedNamePolicy,
    ) -> Self {
        Self::Column(FieldColumn {
            common: ColumnCommon::new(name, leaf_spec, nested_name_policy),
            source,
            wrapper_shape,
        })
    }

    pub(crate) const fn tuple(
        source: FieldSource,
        wrapper_shape: WrapperShape,
        elements: NonEmpty<TupleNode>,
    ) -> Self {
        Self::Tuple(TupleField {
            source,
            wrapper_shape,
            elements,
        })
    }

    pub fn visit_terminal_columns(&self, visitor: &mut impl FnMut(TerminalColumnRef<'_>)) {
        match self {
            Self::Column(column) => visitor(TerminalColumnRef {
                common: &column.common,
                vec_depth: column.wrapper_shape.vec_depth(),
            }),
            Self::Tuple(tuple) => {
                let parent_vec_depth = tuple.wrapper_shape.vec_depth();
                for element in tuple.elements.iter() {
                    element.visit_terminal_columns(parent_vec_depth, visitor);
                }
            }
        }
    }

    pub fn terminal_column_count(&self) -> usize {
        let mut count = 0;
        self.visit_terminal_columns(&mut |_| count += 1);
        count
    }

    pub fn has_vec_shape(&self) -> bool {
        match self {
            Self::Column(column) => column.wrapper_shape.vec_depth() > 0,
            Self::Tuple(tuple) => {
                tuple.wrapper_shape.vec_depth() > 0
                    || tuple.elements.iter().any(TupleNode::has_vec_shape)
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct TerminalColumnRef<'a> {
    common: &'a ColumnCommon,
    vec_depth: usize,
}

impl<'a> TerminalColumnRef<'a> {
    pub fn name(self) -> &'a str {
        self.common.name()
    }

    pub const fn leaf_spec(self) -> &'a TerminalLeafSpec {
        self.common.leaf_spec()
    }

    pub const fn nested_name_policy(self) -> &'a NestedNamePolicy {
        self.common.nested_name_policy()
    }

    pub const fn vec_depth(self) -> usize {
        self.vec_depth
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnCommon {
    name: String,
    leaf_spec: TerminalLeafSpec,
    nested_name_policy: NestedNamePolicy,
}

impl ColumnCommon {
    const fn new(
        name: String,
        leaf_spec: TerminalLeafSpec,
        nested_name_policy: NestedNamePolicy,
    ) -> Self {
        Self {
            name,
            leaf_spec,
            nested_name_policy,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn leaf_spec(&self) -> &TerminalLeafSpec {
        &self.leaf_spec
    }

    pub const fn nested_name_policy(&self) -> &NestedNamePolicy {
        &self.nested_name_policy
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldColumn {
    common: ColumnCommon,
    source: FieldSource,
    wrapper_shape: WrapperShape,
}

impl FieldColumn {
    pub fn name(&self) -> &str {
        self.common.name()
    }

    pub const fn leaf_spec(&self) -> &TerminalLeafSpec {
        self.common.leaf_spec()
    }

    pub const fn nested_name_policy(&self) -> &NestedNamePolicy {
        self.common.nested_name_policy()
    }

    pub const fn source(&self) -> &FieldSource {
        &self.source
    }

    pub const fn wrapper_shape(&self) -> &WrapperShape {
        &self.wrapper_shape
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleField {
    source: FieldSource,
    wrapper_shape: WrapperShape,
    elements: NonEmpty<TupleNode>,
}

impl TupleField {
    pub const fn source(&self) -> &FieldSource {
        &self.source
    }

    pub const fn wrapper_shape(&self) -> &WrapperShape {
        &self.wrapper_shape
    }

    pub const fn elements(&self) -> &NonEmpty<TupleNode> {
        &self.elements
    }
}

/// One tuple element. Nested tuples remain nodes rather than being flattened
/// into independent terminal columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleNode {
    step: TupleProjectionStep,
    wrapper_shape: WrapperShape,
    kind: TupleNodeKind,
}

impl TupleNode {
    pub(crate) fn leaf(
        name: String,
        step: TupleProjectionStep,
        wrapper_shape: WrapperShape,
        leaf_spec: TerminalLeafSpec,
    ) -> Self {
        Self {
            step,
            wrapper_shape,
            kind: TupleNodeKind::Leaf(Box::new(ColumnCommon::new(
                name,
                leaf_spec,
                NestedNamePolicy::Field,
            ))),
        }
    }

    pub(crate) fn tuple(
        step: TupleProjectionStep,
        wrapper_shape: WrapperShape,
        elements: NonEmpty<Self>,
    ) -> Self {
        Self {
            step,
            wrapper_shape,
            kind: TupleNodeKind::Tuple(Box::new(elements)),
        }
    }

    pub const fn step(&self) -> TupleProjectionStep {
        self.step
    }

    pub const fn wrapper_shape(&self) -> &WrapperShape {
        &self.wrapper_shape
    }

    pub const fn kind(&self) -> &TupleNodeKind {
        &self.kind
    }

    fn visit_terminal_columns(
        &self,
        parent_vec_depth: usize,
        visitor: &mut impl FnMut(TerminalColumnRef<'_>),
    ) {
        let vec_depth = parent_vec_depth + self.wrapper_shape.vec_depth();
        match &self.kind {
            TupleNodeKind::Leaf(common) => visitor(TerminalColumnRef {
                common: common.as_ref(),
                vec_depth,
            }),
            TupleNodeKind::Tuple(elements) => {
                for element in elements.iter() {
                    element.visit_terminal_columns(vec_depth, visitor);
                }
            }
        }
    }

    fn has_vec_shape(&self) -> bool {
        self.wrapper_shape.vec_depth() > 0
            || match &self.kind {
                TupleNodeKind::Leaf(_) => false,
                TupleNodeKind::Tuple(elements) => elements.iter().any(Self::has_vec_shape),
            }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TupleNodeKind {
    Leaf(Box<ColumnCommon>),
    Tuple(Box<NonEmpty<TupleNode>>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldSource {
    pub name: Ident,
    pub field_index: Option<usize>,
    pub outer_smart_ptr_depth: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TupleProjectionStep {
    pub index: usize,
    pub outer_smart_ptr_depth: usize,
}
