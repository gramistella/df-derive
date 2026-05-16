//! Per-column codegen entry point. Translates a [`ColumnIR`] into the four
//! pieces of generated code each column contributes to: schema entries,
//! empty-series rows, columnar populator decls/pushes/finishes.
//!
//! The columnar path routes through the encoder IR in
//! [`super::encoder`] for every primitive shape — bare leaves, arbitrary
//! `Option<…<Option<T>>>` stacks, and every vec-bearing wrapper stack.

use crate::ir::{
    ColumnIR, ColumnSource, NestedLeaf, PrimitiveLeaf, ProjectionContext, TerminalLeafRoute,
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

/// Whether a column emission produces schema entries (`(name, dtype)`
/// tuples) or empty-series rows. Both modes iterate the same column set and
/// share the Primitive-vs-Nested classification; only the leaf expression
/// (and the runtime accumulator inside [`super::schema_nested`]) varies.
#[derive(Clone, Copy)]
pub(in crate::codegen) enum EmitMode {
    SchemaEntries,
    EmptyRows,
}

/// Shared column emitter for the schema / empty-rows pair.
fn build_column_entries(
    column: &ColumnIR,
    mode: EmitMode,
    config: &super::MacroConfig,
) -> TokenStream {
    let name = column.name.as_str();
    match (column_leaf_route(column), mode) {
        (TerminalLeafRoute::Nested(nested), EmitMode::SchemaEntries) => {
            let type_path = nested_type_path(nested);
            super::schema_nested::generate_schema_entries_for_struct(
                &type_path,
                &config.traits.to_dataframe,
                name,
                column.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        (TerminalLeafRoute::Nested(nested), EmitMode::EmptyRows) => {
            let type_path = nested_type_path(nested);
            super::schema_nested::nested_empty_series_row(
                &type_path,
                &config.traits.to_dataframe,
                name,
                column.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        (TerminalLeafRoute::Primitive(leaf), EmitMode::SchemaEntries) => {
            let dtype = column_full_dtype(leaf, column, config);
            quote! { ::std::vec![(::std::string::String::from(#name), #dtype)] }
        }
        (TerminalLeafRoute::Primitive(leaf), EmitMode::EmptyRows) => {
            let dtype = column_full_dtype(leaf, column, config);
            let pp = config.external_paths.prelude();
            quote! { ::std::vec![#pp::Series::new_empty(#name.into(), &#dtype).into()] }
        }
    }
}

/// Build the schema entries token expression for one column. Evaluates to a
/// `Vec<(String, DataType)>` at runtime — primitive columns return a
/// one-element vec, nested columns return one entry per inner schema column
/// (with the parent name prefixed).
pub fn build_schema_entries(column: &ColumnIR, config: &super::MacroConfig) -> TokenStream {
    build_column_entries(column, EmitMode::SchemaEntries, config)
}

/// Build the empty-series token expression for one column. Evaluates to a
/// `Vec<Column>` at runtime — primitive columns produce one empty Series,
/// nested columns produce one empty Series per inner schema column.
pub fn build_empty_series(column: &ColumnIR, config: &super::MacroConfig) -> TokenStream {
    build_column_entries(column, EmitMode::EmptyRows, config)
}

fn column_full_dtype(
    leaf: PrimitiveLeaf<'_>,
    column: &ColumnIR,
    config: &super::MacroConfig,
) -> TokenStream {
    super::type_registry::full_dtype(leaf, &column.wrapper_shape, &config.external_paths)
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
    if is_parent_vec_projection(column) {
        return build_parent_vec_projection_emit(column, config, idx);
    }

    if matches!(column.source, ColumnSource::TupleProjection { .. }) {
        return build_projected_standard_emit(column, config, idx, it_ident);
    }

    match column_leaf_route(column) {
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
    column: &ColumnIR,
    config: &super::MacroConfig,
    idx: usize,
    type_path: &TokenStream,
) -> ColumnEmit {
    // The nested encoder paths run their own `for __df_derive_it in items`
    // loops to build their flat ref vec, so the access expression is
    // hard-rooted at the centralized populator-iter ident regardless of the
    // call site's outer-loop binding.
    let inner_it = idents::populator_iter();
    let access = super::access::column_access(column, &inner_it);
    let name = column.name.as_str();
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
    let columnar = encoder::build_nested_encoder(&column.wrapper_shape, &ctx);
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
    column: &ColumnIR,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
    leaf: PrimitiveLeaf<'_>,
) -> ColumnEmit {
    let name = column.name.as_str();
    let access = super::access::column_access(column, it_ident);
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name,
        },
        decimal128_encode_trait: &config.traits.decimal128_encode,
        paths: &config.external_paths,
    };
    let enc = encoder::build_encoder(leaf, &column.wrapper_shape, &leaf_ctx);
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

const fn is_parent_vec_projection(column: &ColumnIR) -> bool {
    matches!(
        column.source,
        ColumnSource::TupleProjection {
            context: ProjectionContext::ParentVec { .. },
            ..
        }
    )
}

fn build_parent_vec_projection_emit(
    column: &ColumnIR,
    config: &super::MacroConfig,
    idx: usize,
) -> ColumnEmit {
    let builder = match column_leaf_route(column) {
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

fn build_projected_standard_emit(
    column: &ColumnIR,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
) -> ColumnEmit {
    let pp = config.external_paths.prelude();
    let access = super::access::column_access(column, it_ident);

    if let TerminalLeafRoute::Nested(nested) = column_leaf_route(column) {
        let type_path = nested_type_path(nested);
        return build_nested_emit_with_access(column, config, idx, &type_path, &access);
    }

    let TerminalLeafRoute::Primitive(leaf) = column_leaf_route(column) else {
        unreachable!("nested route returned above");
    };
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name: &column.name,
        },
        decimal128_encode_trait: &config.traits.decimal128_encode,
        paths: &config.external_paths,
    };
    let enc = encoder::build_encoder_with_option_receiver(
        leaf,
        &column.wrapper_shape,
        &leaf_ctx,
        super::access::column_option_some_receiver(column),
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
            let name = column.name.as_str();
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

const fn column_leaf_route(column: &ColumnIR) -> TerminalLeafRoute<'_> {
    column
        .leaf_spec
        .terminal_route()
        .expect("projected columns never contain tuple leaves")
}

fn build_nested_emit_with_access(
    column: &ColumnIR,
    config: &super::MacroConfig,
    idx: usize,
    type_path: &TokenStream,
    access: &TokenStream,
) -> ColumnEmit {
    let ctx = NestedLeafCtx {
        base: BaseCtx {
            access,
            idx,
            name: &column.name,
        },
        ty: type_path,
        columnar_trait: &config.traits.columnar,
        to_df_trait: &config.traits.to_dataframe,
        paths: &config.external_paths,
    };
    ColumnEmit::WholeColumn {
        builders: vec![encoder::build_nested_encoder(&column.wrapper_shape, &ctx)],
    }
}
