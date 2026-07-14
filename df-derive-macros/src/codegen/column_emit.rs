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
    rows: &Ident,
    sink: &Ident,
) -> ColumnEmit {
    match column {
        ColumnIR::Field(column) => {
            build_field_column_emit(column, config, idx, it_ident, rows, sink)
        }
        ColumnIR::TupleStatic(column) => {
            build_tuple_static_emit(column, config, idx, it_ident, rows, sink)
        }
        ColumnIR::TupleParentOption(column) => {
            build_tuple_parent_option_emit(column, config, idx, it_ident, rows, sink)
        }
        ColumnIR::TupleParentVec(column) => {
            build_parent_vec_projection_emit(column, config, idx, rows, sink)
        }
    }
}

fn build_field_column_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
    rows: &Ident,
    sink: &Ident,
) -> ColumnEmit {
    match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            build_nested_emit(column, config, idx, &type_path, rows, sink)
        }
        TerminalLeafRoute::Primitive(leaf) => {
            build_primitive_emit(column, config, idx, it_ident, leaf, rows, sink)
        }
    }
}

fn build_nested_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    idx: usize,
    type_path: &TokenStream,
    rows: &Ident,
    sink: &Ident,
) -> ColumnEmit {
    // The nested encoder paths run their own `for __df_derive_it in
    // rows.iter().copied()`
    // loops to build their flat ref vec, so the access expression is
    // hard-rooted at the centralized populator-iter ident regardless of the
    // call site's outer-loop binding.
    let inner_it = idents::populator_iter();
    let access = super::source_access::field_column_access(column, &inner_it);
    let name = column.name();
    let ctx = NestedLeafCtx {
        base: BaseCtx {
            access: &access,
            rows,
            sink,
            idx,
            name,
        },
        name_policy: column.nested_name_policy(),
        ty: type_path,
        columnar_trait: &config.runtime.columnar,
        columnar_spec_trait: &config.runtime.columnar_spec,
        paths: &config.external_paths,
    };
    let columnar = encoder::build_nested_encoder(column.wrapper_shape(), &ctx);
    ColumnEmit::WholeColumn {
        builders: vec![columnar],
    }
}

/// Build the columnar emit pieces for a primitive-routed column. `[Vec, ...]`
/// shapes produce `Encoder::Multi` (the encoder packs buffers, the one-pass
/// fill loop, leaf array, list stacking, and the rename + push into one
/// self-contained block). Bare and `[Option]` shapes produce `Encoder::Leaf`
/// with decls + push + finisher split across the three slots.
fn build_primitive_emit(
    column: &FieldColumn,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
    leaf: PrimitiveLeaf<'_>,
    rows: &Ident,
    sink: &Ident,
) -> ColumnEmit {
    let name = column.name();
    let access = super::source_access::field_column_access(column, it_ident);
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            rows,
            sink,
            idx,
            name,
        },
        decimal128_encode_trait: &config.runtime.decimal128_encode,
        paths: &config.external_paths,
    };
    let enc = encoder::build_encoder(leaf, column.wrapper_shape(), &leaf_ctx);
    match enc {
        Encoder::Leaf {
            decls,
            push,
            series,
        } => {
            let builder = quote! {{
                let s = #series;
                #sink.push(s.into())?;
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
    rows: &Ident,
    sink: &Ident,
) -> ColumnEmit {
    let builder = match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            encoder::build_projected_vec_nested(column, &type_path, idx, config, rows, sink)
        }
        TerminalLeafRoute::Primitive(leaf) => {
            encoder::build_projected_vec_primitive(column, leaf, idx, config, rows, sink)
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
    rows: &Ident,
    sink: &Ident,
) -> ColumnEmit {
    let access = super::source_access::tuple_static_access(column, it_ident);
    let base = BaseCtx {
        access: &access,
        rows,
        sink,
        idx,
        name: column.name(),
    };
    build_projected_standard_emit(
        column.leaf_spec(),
        column.wrapper_shape(),
        base,
        None,
        config,
    )
}

fn build_tuple_parent_option_emit(
    column: &TupleParentOptionColumn,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
    rows: &Ident,
    sink: &Ident,
) -> ColumnEmit {
    let access = super::source_access::tuple_parent_option_access(column, it_ident);
    let option_receiver = super::source_access::tuple_parent_option_some_receiver(column);
    let base = BaseCtx {
        access: &access,
        rows,
        sink,
        idx,
        name: column.name(),
    };
    build_projected_standard_emit(
        column.leaf_spec(),
        column.wrapper_shape(),
        base,
        option_receiver,
        config,
    )
}

fn build_projected_standard_emit(
    leaf_spec: &TerminalLeafSpec,
    wrapper_shape: &WrapperShape,
    base: BaseCtx<'_>,
    option_receiver: Option<super::type_registry::PrimitiveExprReceiver>,
    config: &super::MacroConfig,
) -> ColumnEmit {
    let pp = config.external_paths.prelude();

    if let TerminalLeafRoute::Nested(nested) = leaf_spec.route() {
        let type_path = nested_type_path(nested);
        return build_nested_emit_with_access(wrapper_shape, config, &type_path, base);
    }

    let TerminalLeafRoute::Primitive(leaf) = leaf_spec.route() else {
        unreachable!("nested route returned above");
    };
    let name = base.name;
    let idx = base.idx;
    let rows = base.rows;
    let sink = base.sink;
    let leaf_ctx = LeafCtx {
        base,
        decimal128_encode_trait: &config.runtime.decimal128_encode,
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
            quote! {
                {
                    #(#decls)*
                    for #it in #rows.iter().copied() { #push }
                    let #series_local: #pp::Series = #series;
                    let #named = #series_local.with_name(#name.into());
                    #sink.push(#named.into())?;
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
    wrapper_shape: &WrapperShape,
    config: &super::MacroConfig,
    type_path: &TokenStream,
    base: BaseCtx<'_>,
) -> ColumnEmit {
    let name_policy = crate::ir::NestedNamePolicy::Field;
    let ctx = NestedLeafCtx {
        base,
        name_policy: &name_policy,
        ty: type_path,
        columnar_trait: &config.runtime.columnar,
        columnar_spec_trait: &config.runtime.columnar_spec,
        paths: &config.external_paths,
    };
    ColumnEmit::WholeColumn {
        builders: vec![encoder::build_nested_encoder(wrapper_shape, &ctx)],
    }
}
