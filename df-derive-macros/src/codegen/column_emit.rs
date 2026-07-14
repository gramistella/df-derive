//! Per-field encoder dispatch.

use crate::ir::{FieldColumn, FieldPlan, NestedLeaf, PrimitiveLeaf, TerminalLeafRoute};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::encoder::{
    self, BaseCtx, EncodeLifecycle, Encoder, LeafCtx, NestedLeafCtx, idents, struct_type_tokens,
};

pub(in crate::codegen) struct FieldEmit {
    pub decls: Vec<TokenStream>,
    pub push: TokenStream,
    pub builders: Vec<TokenStream>,
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
        row_capacity,
        sink,
    } = params;
    match field {
        FieldPlan::Column(column) => {
            let lifecycle = build_field_column_emit(
                column,
                config,
                ident_scope,
                terminal_start,
                row,
                row_capacity,
                sink,
            );
            FieldEmit {
                decls: lifecycle.decls,
                push: lifecycle.push,
                builders: lifecycle.builders,
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
                    row_capacity,
                    sink,
                },
            );
            FieldEmit {
                decls: tuple.lifecycle.decls,
                push: tuple.lifecycle.push,
                builders: tuple.lifecycle.builders,
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
) -> EncodeLifecycle {
    match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            build_nested_emit(column, config, idx, row, &type_path, row_capacity, sink)
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
    idx: usize,
    row: &Ident,
    type_path: &TokenStream,
    row_capacity: &Ident,
    sink: &Ident,
) -> EncodeLifecycle {
    let access = super::source_access::field_column_access(column, row);
    let ctx = NestedLeafCtx {
        base: BaseCtx {
            access: &access,
            row_capacity,
            sink,
            idx,
            name: column.name(),
        },
        name_policy: column.nested_name_policy(),
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
) -> EncodeLifecycle {
    let name = column.name();
    let access = super::source_access::field_column_access(column, row);
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            row_capacity,
            sink,
            idx,
            name,
        },
        decimal128_encode_trait: &config.runtime.decimal128_encode,
        paths: &config.external_paths,
    };
    match encoder::build_encoder(leaf, column.wrapper_shape(), &leaf_ctx) {
        Encoder::Leaf {
            decls,
            push,
            series,
        } => {
            let output_series = idents::field_output_series(ident_scope);
            EncodeLifecycle {
                decls,
                push,
                builders: vec![quote! {{
                    let #output_series = #series;
                    #sink.push(#output_series.into())?;
                }}],
            }
        }
        Encoder::Multi(mut lifecycle) => {
            let series = idents::vec_field_series(idx);
            let named = idents::field_named_series();
            lifecycle.builders.push(quote! {
                {
                    let #named = #series.with_name(#name.into());
                    #sink.push(#named.into())?;
                }
            });
            lifecycle
        }
    }
}
