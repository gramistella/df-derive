use syn::Ident;

use super::{AccessChain, NonEmpty, TerminalLeafSpec, VecLayers, WrapperShape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnIR {
    Field(FieldColumn),
    TupleStatic(TupleStaticColumn),
    TupleParentOption(TupleParentOptionColumn),
    TupleParentVec(TupleParentVecColumn),
}

impl ColumnIR {
    pub(crate) const fn field(
        name: String,
        source: FieldSource,
        leaf_spec: TerminalLeafSpec,
        wrapper_shape: WrapperShape,
    ) -> Self {
        Self::Field(FieldColumn {
            common: ColumnCommon::new(name, leaf_spec),
            source,
            wrapper_shape,
        })
    }

    pub(crate) const fn tuple_static(
        name: String,
        root: FieldSource,
        path: TupleProjectionPath,
        leaf_spec: TerminalLeafSpec,
        wrapper_shape: WrapperShape,
    ) -> Self {
        Self::TupleStatic(TupleStaticColumn {
            common: ColumnCommon::new(name, leaf_spec),
            root,
            path,
            wrapper_shape,
        })
    }

    pub(crate) const fn tuple_parent_option(
        name: String,
        root: FieldSource,
        path: TupleProjectionPath,
        parent_access: AccessChain,
        leaf_spec: TerminalLeafSpec,
        wrapper_shape: WrapperShape,
    ) -> Self {
        Self::TupleParentOption(TupleParentOptionColumn {
            common: ColumnCommon::new(name, leaf_spec),
            root,
            path,
            parent_access,
            wrapper_shape,
        })
    }

    pub(crate) const fn tuple_parent_vec(
        name: String,
        root: FieldSource,
        terminal_step: TupleProjectionStep,
        projection_layer: usize,
        parent_inner_access: AccessChain,
        leaf_spec: TerminalLeafSpec,
        wrapper_shape: VecLayers,
    ) -> Self {
        Self::TupleParentVec(TupleParentVecColumn {
            common: ColumnCommon::new(name, leaf_spec),
            root,
            terminal_step,
            projection_layer,
            parent_inner_access,
            wrapper_shape,
        })
    }

    pub fn name(&self) -> &str {
        self.common().name()
    }

    pub const fn leaf_spec(&self) -> &TerminalLeafSpec {
        self.common().leaf_spec()
    }

    pub const fn vec_depth(&self) -> usize {
        match self {
            Self::Field(column) => column.wrapper_shape.vec_depth(),
            Self::TupleStatic(column) => column.wrapper_shape.vec_depth(),
            Self::TupleParentOption(column) => column.wrapper_shape.vec_depth(),
            Self::TupleParentVec(column) => column.wrapper_shape.depth(),
        }
    }

    const fn common(&self) -> &ColumnCommon {
        match self {
            Self::Field(column) => &column.common,
            Self::TupleStatic(column) => &column.common,
            Self::TupleParentOption(column) => &column.common,
            Self::TupleParentVec(column) => &column.common,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnCommon {
    name: String,
    leaf_spec: TerminalLeafSpec,
}

impl ColumnCommon {
    const fn new(name: String, leaf_spec: TerminalLeafSpec) -> Self {
        Self { name, leaf_spec }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn leaf_spec(&self) -> &TerminalLeafSpec {
        &self.leaf_spec
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

    pub const fn source(&self) -> &FieldSource {
        &self.source
    }

    pub const fn wrapper_shape(&self) -> &WrapperShape {
        &self.wrapper_shape
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleStaticColumn {
    common: ColumnCommon,
    root: FieldSource,
    path: TupleProjectionPath,
    wrapper_shape: WrapperShape,
}

impl TupleStaticColumn {
    pub fn name(&self) -> &str {
        self.common.name()
    }

    pub const fn leaf_spec(&self) -> &TerminalLeafSpec {
        self.common.leaf_spec()
    }

    pub const fn root(&self) -> &FieldSource {
        &self.root
    }

    pub const fn path(&self) -> &TupleProjectionPath {
        &self.path
    }

    pub const fn wrapper_shape(&self) -> &WrapperShape {
        &self.wrapper_shape
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleParentOptionColumn {
    common: ColumnCommon,
    root: FieldSource,
    path: TupleProjectionPath,
    parent_access: AccessChain,
    wrapper_shape: WrapperShape,
}

impl TupleParentOptionColumn {
    pub fn name(&self) -> &str {
        self.common.name()
    }

    pub const fn leaf_spec(&self) -> &TerminalLeafSpec {
        self.common.leaf_spec()
    }

    pub const fn root(&self) -> &FieldSource {
        &self.root
    }

    pub const fn path(&self) -> &TupleProjectionPath {
        &self.path
    }

    pub const fn parent_access(&self) -> &AccessChain {
        &self.parent_access
    }

    pub const fn wrapper_shape(&self) -> &WrapperShape {
        &self.wrapper_shape
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleParentVecColumn {
    common: ColumnCommon,
    root: FieldSource,
    terminal_step: TupleProjectionStep,
    projection_layer: usize,
    parent_inner_access: AccessChain,
    wrapper_shape: VecLayers,
}

impl TupleParentVecColumn {
    pub fn name(&self) -> &str {
        self.common.name()
    }

    pub const fn leaf_spec(&self) -> &TerminalLeafSpec {
        self.common.leaf_spec()
    }

    pub const fn root(&self) -> &FieldSource {
        &self.root
    }

    pub const fn terminal_step(&self) -> TupleProjectionStep {
        self.terminal_step
    }

    pub const fn projection_layer(&self) -> usize {
        self.projection_layer
    }

    pub const fn parent_inner_access(&self) -> &AccessChain {
        &self.parent_inner_access
    }

    pub const fn wrapper_shape(&self) -> &VecLayers {
        &self.wrapper_shape
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleProjectionPath {
    steps: NonEmpty<TupleProjectionStep>,
}

impl TupleProjectionPath {
    pub(crate) fn from_vec(steps: Vec<TupleProjectionStep>) -> Option<Self> {
        NonEmpty::from_vec(steps).map(|steps| Self { steps })
    }

    pub fn iter(&self) -> impl Iterator<Item = &TupleProjectionStep> {
        self.steps.iter()
    }
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
