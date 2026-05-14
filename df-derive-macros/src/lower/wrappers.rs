use crate::ir::{
    AccessChain, AccessStep, LeafShape, NonEmpty, VecLayerSpec, VecLayers, WrapperShape,
};
use crate::type_analysis::RawWrapper;

/// Normalize the raw outer-to-inner `RawWrapper` sequence into a
/// `WrapperShape` the encoder consumes directly. `Option` and smart-pointer
/// steps are retained as an `AccessChain` at each wrapper boundary: above
/// each `Vec`, immediately surrounding the leaf, or for the leaf-only path.
/// Polars folds consecutive `Option`s into a single validity bit per
/// position, so the count is also cached to choose the direct single-Option
/// path versus the collapsed multi-Option path.
pub fn normalize_wrappers(wrappers: &[RawWrapper]) -> WrapperShape {
    let mut layers: Vec<VecLayerSpec> = Vec::new();
    let mut pending_access = AccessChain::empty();
    for w in wrappers {
        match w {
            RawWrapper::Option => {
                pending_access.steps.push(AccessStep::Option);
            }
            RawWrapper::SmartPtr => {
                pending_access.steps.push(AccessStep::SmartPtr);
            }
            RawWrapper::Vec => {
                let option_layers_above = pending_access.option_layers();
                layers.push(VecLayerSpec {
                    option_layers_above,
                    access: std::mem::take(&mut pending_access),
                });
            }
        }
    }
    let Some(layers) = NonEmpty::from_vec(layers) else {
        return WrapperShape::Leaf(LeafShape::from_option_access(
            pending_access.option_layers(),
            pending_access,
        ));
    };
    WrapperShape::Vec(VecLayers {
        layers,
        inner_option_layers: pending_access.option_layers(),
        inner_access: pending_access,
    })
}
