use proc_macro2::TokenStream;

use crate::codegen::encode_plan::SeriesPlan;
use crate::codegen::external_paths::ExternalPaths;
use crate::codegen::planner::PrimitiveListPolicy;
use crate::ir::{PrimitiveLeaf, WrapperShape};

use super::idents::GeneratedIdentScope;
use super::{leaf, option, vec};

pub struct BaseCtx<'a> {
    pub access: &'a TokenStream,
    pub row_capacity: &'a syn::Ident,
    pub sink: &'a syn::Ident,
    pub idx: usize,
    pub name: &'a str,
}

#[derive(Clone, Copy)]
pub struct RowReplay<'a> {
    pub row: &'a syn::Ident,
    pub replay: &'a TokenStream,
}

/// Relates one primitive-leaf push to the source iterator.
///
/// Only row-aligned leaves may seed validity from an exact iterator size
/// hint. Leaves below a list traversal have an independent flattened length
/// and must grow validity as values are observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafCardinality {
    InputRows,
    Dynamic,
}

pub struct LeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub primitive_list_plan: Option<PrimitiveListPolicy>,
    pub row_replay: Option<RowReplay<'a>>,
    pub cardinality: LeafCardinality,
    pub ident_scope: GeneratedIdentScope<'a>,
    pub input_rows_exact: &'a syn::Ident,
    pub decimal128_encode_trait: &'a syn::Path,
    pub paths: &'a ExternalPaths,
}

pub fn build_encoder(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    ctx: &LeafCtx<'_>,
) -> SeriesPlan {
    build_encoder_with_option_receiver(leaf, wrapper, ctx, None)
}

pub(in crate::codegen) fn build_encoder_with_option_receiver(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    ctx: &LeafCtx<'_>,
    option_some_receiver: Option<crate::codegen::type_registry::PrimitiveExprReceiver>,
) -> SeriesPlan {
    match wrapper {
        WrapperShape::Leaf(shape) if shape.is_bare() => {
            let leaf::LeafArm {
                decls,
                push,
                series,
            } = vec::build_leaf(leaf, ctx, leaf::LeafArmKind::Bare);
            SeriesPlan::leaf(decls, push, series)
        }
        WrapperShape::Leaf(shape) if shape.access().is_single_plain_option() => {
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
            SeriesPlan::leaf(decls, push, series)
        }
        WrapperShape::Leaf(shape) => option::wrap_option_access_chain_primitive(
            leaf,
            ctx,
            shape.access(),
            shape.access().option_layers(),
        ),
        WrapperShape::Vec(vec_layers) => {
            let Some(plan) = ctx.primitive_list_plan else {
                unreachable!("primitive-list policy must be selected before lowering")
            };
            vec::try_build_vec_encoder(leaf, ctx, vec_layers, plan)
        }
    }
}
