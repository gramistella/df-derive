//! Per-column encoder dispatch.

use crate::ir::{
    ColumnIR, FieldColumn, NestedLeaf, PrimitiveLeaf, TerminalLeafRoute, TerminalLeafSpec,
    TupleParentOptionColumn, TupleParentVecColumn, TupleStaticColumn, WrapperShape,
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::encoder::{self, BaseCtx, Encoder, LeafCtx, NestedLeafCtx, idents, struct_type_tokens};

/// Per-column emission mode.
///
/// Row-wise columns split setup, per-row push, and final builder materialization
/// across the surrounding columnar pipeline. Whole-column emitters build their
/// columns in self-contained post-loop blocks.
pub(in crate::codegen) enum ColumnEmit {
    RowWise {
        decls: Vec<TokenStream>,
        push: TokenStream,
        builders: Vec<TokenStream>,
    },
    WholeColumn {
        builders: Vec<TokenStream>,
    },
}

fn nested_type_path(nested: NestedLeaf<'_>) -> TokenStream {
    match nested {
        NestedLeaf::Struct(ty) => struct_type_tokens(ty),
        NestedLeaf::Generic(id) => quote! { #id },
    }
}

/// Build the columnar emit pieces for one column. Routes every primitive
/// shape through the encoder IR, and every nested-struct/generic column
/// through the encoder's nested path (which covers every wrapper stack).
pub fn build_column_emit(
    column: &ColumnIR,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
) -> ColumnEmit {
    match column {
        ColumnIR::Field(column) => build_field_column_emit(column, config, idx, it_ident),
        ColumnIR::TupleStatic(column) => build_tuple_static_emit(column, config, idx, it_ident),
        ColumnIR::TupleParentOption(column) => {
            build_tuple_parent_option_emit(column, config, idx, it_ident)
        }
        ColumnIR::TupleParentVec(column) => build_parent_vec_projection_emit(column, config, idx),
    }
}

fn build_field_column_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
) -> ColumnEmit {
    match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            build_nested_emit(column, config, idx, &type_path)
        }
        TerminalLeafRoute::Primitive(leaf) => {
            build_primitive_emit(column, config, idx, it_ident, leaf)
        }
    }
}

fn build_nested_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    idx: usize,
    type_path: &TokenStream,
) -> ColumnEmit {
    // The nested encoder paths run their own `for __df_derive_it in items`
    // loops to build their flat ref vec, so the access expression is
    // hard-rooted at the centralized populator-iter ident regardless of the
    // call site's outer-loop binding.
    let inner_it = idents::populator_iter();
    let access = super::source_access::field_column_access(column, &inner_it);
    let name = column.name();
    let ctx = NestedLeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name,
        },
        ty: type_path,
        columnar_trait: &config.traits.columnar,
        to_df_trait: &config.traits.to_dataframe,
        paths: &config.external_paths,
    };
    let columnar = encoder::build_nested_encoder(column.wrapper_shape(), &ctx);
    ColumnEmit::WholeColumn {
        builders: vec![columnar],
    }
}

/// Build the columnar emit pieces for a primitive-routed column. `[Vec, ...]`
/// shapes produce `Encoder::Multi` (the encoder packs precount, buffers,
/// fill loop, leaf array, list stacking, and the rename + push into one
/// self-contained block). Bare and `[Option]` shapes produce `Encoder::Leaf`
/// with decls + push + finisher split across the three slots.
fn build_primitive_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
    leaf: PrimitiveLeaf<'_>,
) -> ColumnEmit {
    let name = column.name();
    let access = super::source_access::field_column_access(column, it_ident);
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name,
        },
        decimal128_encode_trait: &config.traits.decimal128_encode,
        paths: &config.external_paths,
    };
    let enc = encoder::build_encoder(leaf, column.wrapper_shape(), &leaf_ctx);
    match enc {
        Encoder::Leaf {
            decls,
            push,
            series,
        } => {
            let columns = idents::columns();
            let builder = quote! {{
                let s = #series;
                #columns.push(s.into());
            }};
            ColumnEmit::RowWise {
                decls,
                push,
                builders: vec![builder],
            }
        }
        Encoder::Multi { columnar } => ColumnEmit::WholeColumn {
            builders: vec![columnar],
        },
    }
}

fn build_parent_vec_projection_emit(
    column: &TupleParentVecColumn,
    config: &super::MacroConfig,
    idx: usize,
) -> ColumnEmit {
    let builder = match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            encoder::build_projected_vec_nested(column, &type_path, idx, config)
        }
        TerminalLeafRoute::Primitive(leaf) => {
            encoder::build_projected_vec_primitive(column, leaf, idx, config)
        }
    };
    ColumnEmit::WholeColumn {
        builders: vec![builder],
    }
}

fn build_tuple_static_emit(
    column: &TupleStaticColumn,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
) -> ColumnEmit {
    let access = super::source_access::tuple_static_access(column, it_ident);
    build_projected_standard_emit(
        column.name(),
        column.leaf_spec(),
        column.wrapper_shape(),
        &access,
        None,
        config,
        idx,
    )
}

fn build_tuple_parent_option_emit(
    column: &TupleParentOptionColumn,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
) -> ColumnEmit {
    let access = super::source_access::tuple_parent_option_access(column, it_ident);
    let option_receiver = super::source_access::tuple_parent_option_some_receiver(column);
    build_projected_standard_emit(
        column.name(),
        column.leaf_spec(),
        column.wrapper_shape(),
        &access,
        option_receiver,
        config,
        idx,
    )
}

fn build_projected_standard_emit(
    name: &str,
    leaf_spec: &TerminalLeafSpec,
    wrapper_shape: &WrapperShape,
    access: &TokenStream,
    option_receiver: Option<super::type_registry::PrimitiveExprReceiver>,
    config: &super::MacroConfig,
    idx: usize,
) -> ColumnEmit {
    let pp = config.external_paths.prelude();

    if let TerminalLeafRoute::Nested(nested) = leaf_spec.route() {
        let type_path = nested_type_path(nested);
        return build_nested_emit_with_access(name, wrapper_shape, config, idx, &type_path, access);
    }

    let TerminalLeafRoute::Primitive(leaf) = leaf_spec.route() else {
        unreachable!("nested route returned above");
    };
    let leaf_ctx = LeafCtx {
        base: BaseCtx { access, idx, name },
        decimal128_encode_trait: &config.traits.decimal128_encode,
        paths: &config.external_paths,
    };
    let enc = encoder::build_encoder_with_option_receiver(
        leaf,
        wrapper_shape,
        &leaf_ctx,
        option_receiver,
    );
    let builder = match enc {
        Encoder::Leaf {
            decls,
            push,
            series,
        } => {
            let it = idents::populator_iter();
            let named = idents::field_named_series();
            let series_local = idents::vec_field_series(idx);
            let columns = idents::columns();
            quote! {
                {
                    #(#decls)*
                    for #it in items { #push }
                    let #series_local: #pp::Series = #series;
                    let #named = #series_local.with_name(#name.into());
                    #columns.push(#named.into());
                }
            }
        }
        Encoder::Multi { columnar } => columnar,
    };
    ColumnEmit::WholeColumn {
        builders: vec![builder],
    }
}

fn build_nested_emit_with_access(
    name: &str,
    wrapper_shape: &WrapperShape,
    config: &super::MacroConfig,
    idx: usize,
    type_path: &TokenStream,
    access: &TokenStream,
) -> ColumnEmit {
    let ctx = NestedLeafCtx {
        base: BaseCtx { access, idx, name },
        ty: type_path,
        columnar_trait: &config.traits.columnar,
        to_df_trait: &config.traits.to_dataframe,
        paths: &config.external_paths,
    };
    ColumnEmit::WholeColumn {
        builders: vec![encoder::build_nested_encoder(wrapper_shape, &ctx)],
    }
}
