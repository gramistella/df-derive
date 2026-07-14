use super::{AccessChain, NonEmpty};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafShape {
    access: AccessChain,
}

impl LeafShape {
    pub fn from_access(access: AccessChain) -> Self {
        assert!(
            access.is_empty() || access.has_option(),
            "non-empty leaf access must carry optionality"
        );
        Self { access }
    }

    pub const fn access(&self) -> &AccessChain {
        &self.access
    }

    pub const fn is_bare(&self) -> bool {
        self.access.is_empty()
    }
}

/// Polars folds consecutive `Option`s at a list level into one validity bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecLayerSpec {
    pub access: AccessChain,
}

impl VecLayerSpec {
    pub fn has_outer_validity(&self) -> bool {
        self.access.has_option()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecLayers {
    pub layers: NonEmpty<VecLayerSpec>,
    pub inner_access: AccessChain,
}

impl VecLayers {
    pub const fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn has_inner_option(&self) -> bool {
        self.inner_access.has_option()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WrapperShape {
    Leaf(LeafShape),
    Vec(VecLayers),
}

impl WrapperShape {
    pub const fn vec_depth(&self) -> usize {
        match self {
            Self::Leaf(_) => 0,
            Self::Vec(v) => v.depth(),
        }
    }
}
