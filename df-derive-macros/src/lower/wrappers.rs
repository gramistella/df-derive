use crate::ir::{
    AccessChain, AccessStep, LeafShape, NonEmpty, VecLayerSpec, VecLayers, WrapperShape,
};
use crate::type_analysis::RawWrapper;

/// Normalize the raw outer-to-inner `RawWrapper` sequence into a
/// `WrapperShape` the encoder consumes directly. `Option` and smart-pointer
/// steps are retained as an `AccessChain` at each wrapper boundary: above
/// each `Vec`, immediately surrounding the leaf, or for the leaf-only path.
/// Polars folds consecutive `Option`s into a single validity bit per
/// position. Codegen derives that fact from each boundary's access chain.
pub fn normalize_wrappers(wrappers: &[RawWrapper]) -> WrapperShape {
    let mut layers: Vec<VecLayerSpec> = Vec::new();
    let mut pending_access = AccessChain::empty();
    for w in wrappers {
        match w {
            RawWrapper::Option => {
                pending_access.push(AccessStep::Option);
            }
            RawWrapper::SmartPtr => {
                pending_access.push(AccessStep::SmartPtr);
            }
            RawWrapper::Vec => {
                layers.push(VecLayerSpec {
                    access: std::mem::take(&mut pending_access),
                });
            }
        }
    }
    let Some(layers) = NonEmpty::from_vec(layers) else {
        return WrapperShape::Leaf(LeafShape::from_access(pending_access));
    };
    WrapperShape::Vec(VecLayers {
        layers,
        inner_access: pending_access,
    })
}
