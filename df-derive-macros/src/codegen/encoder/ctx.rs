use proc_macro2::TokenStream;

use crate::codegen::external_paths::ExternalPaths;
use crate::ir::{PrimitiveLeaf, WrapperShape};

use super::idents::GeneratedIdentScope;
use super::{leaf, option, vec};

pub enum Encoder {
    Leaf {
        decls: Vec<TokenStream>,
        push: TokenStream,
        series: TokenStream,
    },
    Multi(EncodeLifecycle),
}

/// The three phases every multi-value encoder contributes to the enclosing
/// one-shot row pass.
///
/// Declarations run before the shared row loop, `push` runs once for the
/// current row, and builders materialize checked columns after the iterator
/// has been exhausted.
pub struct EncodeLifecycle {
    pub decls: Vec<TokenStream>,
    pub push: TokenStream,
    pub builders: Vec<TokenStream>,
}

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
    pub rows: &'a syn::Ident,
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
        WrapperShape::Leaf(shape) if shape.is_bare() => {
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
            Encoder::Leaf {
                decls,
                push,
                series,
            }
        }
        WrapperShape::Leaf(shape) => option::wrap_option_access_chain_primitive(
            leaf,
            ctx,
            shape.access(),
            shape.access().option_layers(),
        ),
        WrapperShape::Vec(vec_layers) => vec::try_build_vec_encoder(leaf, ctx, vec_layers),
    }
}
