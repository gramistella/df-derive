//! Per-field encoder dispatch.

use crate::ir::{
    FieldColumn, FieldPlan, NestedLeaf, PrimitiveLeaf, TerminalLeafRoute, WrapperShape,
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::encode_plan::{EmitOp, EncodePlan, FinishGroup, SeriesPlan};
use super::encoder::{
    self, BaseCtx, LeafCardinality, LeafCtx, NestedLeafCtx, idents, struct_type_tokens,
};

pub(in crate::codegen) struct FieldEmit {
    pub plan: EncodePlan,
    pub terminal_count: usize,
    pub group_count: usize,
}

#[derive(Clone, Copy)]
pub(in crate::codegen) struct FieldEmitParams<'a> {
    pub config: &'a super::MacroConfig,
    pub ident_scope: idents::GeneratedIdentScope<'a>,
    pub terminal_start: usize,
    pub group_start: usize,
    pub row: &'a Ident,
    pub replay: &'a TokenStream,
    pub static_tuple_plan: super::planner::StaticTuplePlan,
    pub row_capacity: &'a Ident,
    pub sink: &'a Ident,
}

fn nested_type_path(nested: NestedLeaf<'_>) -> TokenStream {
    match nested {
        NestedLeaf::Struct(ty) => struct_type_tokens(ty),
        NestedLeaf::Generic(id) => quote! { #id },
    }
}

pub(in crate::codegen) fn build_field_emit(
    field: &FieldPlan,
    params: FieldEmitParams<'_>,
) -> FieldEmit {
    let FieldEmitParams {
        config,
        ident_scope,
        terminal_start,
        group_start,
        row,
        replay,
        static_tuple_plan,
        row_capacity,
        sink,
    } = params;
    match field {
        FieldPlan::Column(column) => {
            let plan = build_field_column_emit(
                column,
                config,
                ident_scope,
                terminal_start,
                row,
                row_capacity,
                sink,
            );
            FieldEmit {
                plan,
                terminal_count: 1,
                group_count: 0,
            }
        }
        FieldPlan::Tuple(tuple) => {
            let tuple = encoder::build_tuple_field_emit(
                tuple,
                encoder::TupleFieldEmitParams {
                    config,
                    ident_scope,
                    terminal_start,
                    group_start,
                    row,
                    replay,
                    static_tuple_plan,
                    row_capacity,
                    sink,
                },
            );
            FieldEmit {
                plan: tuple.plan,
                terminal_count: tuple.terminal_count,
                group_count: tuple.group_count,
            }
        }
    }
}

fn build_field_column_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    row: &Ident,
    row_capacity: &Ident,
    sink: &Ident,
) -> EncodePlan {
    match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            build_nested_emit(
                column,
                config,
                ident_scope,
                idx,
                row,
                &type_path,
                row_capacity,
                sink,
            )
        }
        TerminalLeafRoute::Primitive(leaf) => build_primitive_emit(
            column,
            config,
            ident_scope,
            idx,
            row,
            leaf,
            row_capacity,
            sink,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_nested_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    row: &Ident,
    type_path: &TokenStream,
    row_capacity: &Ident,
    sink: &Ident,
) -> EncodePlan {
    let access = super::source_access::field_column_access(column, row);
    let ctx = NestedLeafCtx {
        base: BaseCtx {
            access: &access,
            row_capacity,
            idx,
        },
        ident_scope,
        sink,
        ty: type_path,
        columnar_trait: &config.runtime.columnar,
        columnar_spec_trait: &config.runtime.columnar_spec,
        paths: &config.external_paths,
    };
    encoder::build_nested_encoder(column.wrapper_shape(), &ctx)
}

#[allow(clippy::too_many_arguments)]
fn build_primitive_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    row: &Ident,
    leaf: PrimitiveLeaf<'_>,
    row_capacity: &Ident,
    sink: &Ident,
) -> EncodePlan {
    let access = super::source_access::field_column_access(column, row);
    let input_rows_exact = idents::input_rows_exact(ident_scope);
    let output_slot = idents::output_slot(ident_scope, idx);
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            row_capacity,
            idx,
        },
        materialization: encoder::MaterializationTarget::schema_slot(&output_slot),
        primitive_list_plan: super::planner::PrimitiveListPolicy::for_wrapper(
            leaf,
            column.wrapper_shape(),
        ),
        cardinality: match column.wrapper_shape() {
            WrapperShape::Leaf(_) => LeafCardinality::InputRows,
            WrapperShape::Vec(_) => LeafCardinality::Dynamic,
        },
        ident_scope,
        input_rows_exact: &input_rows_exact,
        decimal128_encode_trait: &config.runtime.decimal128_encode,
        encode_support: &config.runtime.encode_support,
        paths: &config.external_paths,
    };
    let SeriesPlan {
        init,
        scan,
        post_scan,
        materialize,
        output_slot,
    } = encoder::build_encoder(leaf, column.wrapper_shape(), &leaf_ctx);
    let output_series = idents::field_output_series(ident_scope);
    let finish = vec![FinishGroup::scoped(
        post_scan,
        vec![EmitOp::new(quote! {
            let #output_slot = #sink.next_slot()?;
            let #output_series = #materialize;
            #output_slot.commit(#output_series.into())?;
        })],
    )];
    EncodePlan::new(init, vec![scan], finish)
}
