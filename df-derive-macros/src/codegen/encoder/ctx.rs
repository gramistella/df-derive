use proc_macro2::TokenStream;
use quote::quote;

use crate::codegen::encode_plan::SeriesPlan;
use crate::codegen::external_paths::ExternalPaths;
use crate::codegen::planner::PrimitiveListPolicy;
use crate::ir::{PrimitiveLeaf, WrapperShape};

use super::idents::GeneratedIdentScope;
use super::{leaf, option, vec};

pub struct BaseCtx<'a> {
    pub access: &'a TokenStream,
    pub row_capacity: &'a syn::Ident,
    pub idx: usize,
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

/// How a primitive Series is materialized before its final schema commit.
///
/// A terminal without an enclosing tuple-list prefix can use its schema slot
/// immediately. A prefixed terminal first builds an intermediate Series, then
/// wraps it with schema-slot metadata before committing the final output.
#[derive(Clone)]
pub(in crate::codegen) enum MaterializationTarget {
    SchemaSlot(syn::Ident),
    Intermediate {
        name: String,
        output_slot: syn::Ident,
    },
}

impl MaterializationTarget {
    pub(in crate::codegen) fn schema_slot(output_slot: &syn::Ident) -> Self {
        Self::SchemaSlot(output_slot.clone())
    }

    pub(in crate::codegen) fn intermediate(name: &str, output_slot: &syn::Ident) -> Self {
        Self::Intermediate {
            name: name.to_owned(),
            output_slot: output_slot.clone(),
        }
    }

    pub(super) const fn output_slot(&self) -> &syn::Ident {
        match self {
            Self::SchemaSlot(output_slot) | Self::Intermediate { output_slot, .. } => output_slot,
        }
    }

    pub(super) const fn schema_slot_ident(&self) -> Option<&syn::Ident> {
        match self {
            Self::SchemaSlot(output_slot) => Some(output_slot),
            Self::Intermediate { .. } => None,
        }
    }

    pub(super) fn output_name(&self) -> TokenStream {
        match self {
            Self::SchemaSlot(output_slot) => quote! { #output_slot.name().clone() },
            Self::Intermediate { name, .. } => quote! { #name },
        }
    }
}

pub struct LeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub(in crate::codegen) materialization: MaterializationTarget,
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
            SeriesPlan::leaf(
                decls,
                push,
                series,
                ctx.materialization.output_slot().clone(),
            )
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
            SeriesPlan::leaf(
                decls,
                push,
                series,
                ctx.materialization.output_slot().clone(),
            )
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
