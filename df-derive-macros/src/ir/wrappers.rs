use std::num::NonZeroUsize;

use super::{AccessChain, NonEmpty};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeafShape {
    Bare,
    Optional {
        option_layers: NonZeroUsize,
        access: AccessChain,
    },
}

impl LeafShape {
    pub fn from_option_access(option_layers: usize, access: AccessChain) -> Self {
        NonZeroUsize::new(option_layers).map_or(Self::Bare, |option_layers| Self::Optional {
            option_layers,
            access,
        })
    }

    pub const fn option_layers(&self) -> usize {
        match self {
            Self::Bare => 0,
            Self::Optional { option_layers, .. } => option_layers.get(),
        }
    }
}

/// Polars folds consecutive `Option`s at a list level into one validity bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecLayerSpec {
    pub option_layers_above: usize,
    pub access: AccessChain,
}

impl VecLayerSpec {
    pub const fn has_outer_validity(&self) -> bool {
        self.option_layers_above > 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecLayers {
    pub layers: NonEmpty<VecLayerSpec>,
    pub inner_option_layers: usize,
    pub inner_access: AccessChain,
}

impl VecLayers {
    pub const fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn any_outer_validity(&self) -> bool {
        self.layers.iter().any(VecLayerSpec::has_outer_validity)
    }

    pub const fn has_inner_option(&self) -> bool {
        self.inner_option_layers > 0
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
