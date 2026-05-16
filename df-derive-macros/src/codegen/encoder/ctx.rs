use proc_macro2::TokenStream;

use crate::codegen::external_paths::ExternalPaths;
use crate::ir::{LeafShape, PrimitiveLeaf, WrapperShape};

use super::{leaf, option, vec};

pub enum Encoder {
    Leaf {
        decls: Vec<TokenStream>,
        push: TokenStream,
        series: TokenStream,
    },
    Multi {
        columnar: TokenStream,
    },
}

pub struct BaseCtx<'a> {
    pub access: &'a TokenStream,
    pub idx: usize,
    pub name: &'a str,
}

pub struct LeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub decimal128_encode_trait: &'a syn::Path,
    pub paths: &'a ExternalPaths,
}

pub fn build_encoder(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    ctx: &LeafCtx<'_>,
) -> Encoder {
    build_encoder_with_option_receiver(leaf, wrapper, ctx, None)
}

pub(in crate::codegen) fn build_encoder_with_option_receiver(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    ctx: &LeafCtx<'_>,
    option_some_receiver: Option<crate::codegen::type_registry::PrimitiveExprReceiver>,
) -> Encoder {
    match wrapper {
        WrapperShape::Leaf(LeafShape::Bare) => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(leaf, ctx, leaf::LeafArmKind::Bare);
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        WrapperShape::Leaf(LeafShape::Optional {
            option_layers,
            access,
        }) if option_layers.get() == 1 && access.is_single_plain_option() => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(
                leaf,
                ctx,
                leaf::LeafArmKind::Option {
                    some_receiver: option_some_receiver
                        .unwrap_or(crate::codegen::type_registry::PrimitiveExprReceiver::Ref),
                },
            );
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        WrapperShape::Leaf(LeafShape::Optional {
            option_layers,
            access,
        }) => option::wrap_option_access_chain_primitive(leaf, ctx, access, option_layers.get()),
        WrapperShape::Vec(vec_layers) => vec::try_build_vec_encoder(leaf, ctx, vec_layers),
    }
}
