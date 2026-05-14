//! Per-field codegen entry point. Translates a [`FieldIR`] into the four
//! pieces of generated code each field contributes to: schema entries,
//! empty-series rows, columnar populator decls/pushes/finishes.
//!
//! The columnar path routes through the encoder IR in
//! [`super::encoder`] for every primitive shape — bare leaves, arbitrary
//! `Option<…<Option<T>>>` stacks, and every vec-bearing wrapper stack.

use crate::ir::{FieldIR, LeafRoute, NestedLeaf, PrimitiveLeaf};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::encoder::{self, BaseCtx, Encoder, LeafCtx, NestedLeafCtx, idents, struct_type_tokens};

/// Per-field columnar emission mode.
///
/// Row-wise fields split setup, per-row push, and final builder materialization
/// across the surrounding columnar pipeline. Whole-column fields build their
/// columns in self-contained post-loop blocks.
pub(in crate::codegen) enum FieldEmit {
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

/// `<it>.<field>` — the field-access expression rooted at a per-row
/// iterator binding. Used by every emit path that needs to reach into
/// the per-row item.
///
/// Wraps the raw access in `field.outer_smart_ptr_depth` explicit `*`
/// derefs so `Box<isize> as i64` and `match &(box_option) { Some(_) => ... }`
/// see the inner value: numeric `as` casts and pattern positions don't
/// trigger `Deref` autoderef, so the codegen has to peel manually. Inner
/// smart pointers (below a wrapper) are preserved in the normalized wrapper
/// access chains and dereffed at the wrapper boundary where they occur.
fn it_access(field: &FieldIR, it_ident: &Ident) -> TokenStream {
    let raw = field.field_index.map_or_else(
        || {
            let id = &field.name;
            quote! { #it_ident.#id }
        },
        |i| {
            let li = syn::Index::from(i);
            quote! { #it_ident.#li }
        },
    );
    let mut out = raw;
    for _ in 0..field.outer_smart_ptr_depth {
        out = quote! { (*(#out)) };
    }
    out
}

/// Whether a per-field emission produces schema entries (`(name, dtype)`
/// tuples) or empty-series rows. Both modes iterate the same field set and
/// share the Primitive-vs-Nested classification; only the leaf expression
/// (and the runtime accumulator inside [`super::nested`]) varies.
#[derive(Clone, Copy)]
pub(in crate::codegen) enum EmitMode {
    SchemaEntries,
    EmptyRows,
}

/// Shared per-field emitter for the schema / empty-rows pair. Classifies
/// the field once, then dispatches to the matching [`super::nested`]
/// runtime helper for nested fields, or emits a one-element vec literal
/// for primitive fields.
fn build_field_entries(
    field: &FieldIR,
    mode: EmitMode,
    config: &super::MacroConfig,
) -> TokenStream {
    let name = super::helpers::column_name_for_ident(&field.name);
    match (field.leaf_spec.route(), mode) {
        (LeafRoute::Nested(nested), EmitMode::SchemaEntries) => {
            let type_path = nested_type_path(nested);
            super::nested::generate_schema_entries_for_struct(
                &type_path,
                &config.to_dataframe_trait_path,
                &name,
                field.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        (LeafRoute::Nested(nested), EmitMode::EmptyRows) => {
            let type_path = nested_type_path(nested);
            super::nested::nested_empty_series_row(
                &type_path,
                &config.to_dataframe_trait_path,
                &name,
                field.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        (LeafRoute::Primitive(leaf), EmitMode::SchemaEntries) => {
            let dtype = field_full_dtype(leaf, field, config);
            quote! { ::std::vec![(::std::string::String::from(#name), #dtype)] }
        }
        (LeafRoute::Primitive(leaf), EmitMode::EmptyRows) => {
            let dtype = field_full_dtype(leaf, field, config);
            let pp = config.external_paths.prelude();
            quote! { ::std::vec![#pp::Series::new_empty(#name.into(), &#dtype).into()] }
        }
        (LeafRoute::Tuple(elements), _) => {
            super::encoder::build_tuple_field_entries(field, elements, mode, config)
        }
    }
}

/// Build the schema entries token expression for one field. Evaluates to a
/// `Vec<(String, DataType)>` at runtime — primitive fields return a
/// one-element vec, nested fields return one entry per inner schema column
/// (with the parent name prefixed).
pub fn build_schema_entries(field: &FieldIR, config: &super::MacroConfig) -> TokenStream {
    build_field_entries(field, EmitMode::SchemaEntries, config)
}

/// Build the empty-series token expression for one field. Evaluates to a
/// `Vec<Column>` at runtime — primitive fields produce one empty Series,
/// nested fields produce one empty Series per inner schema column.
pub fn build_empty_series(field: &FieldIR, config: &super::MacroConfig) -> TokenStream {
    build_field_entries(field, EmitMode::EmptyRows, config)
}

fn field_full_dtype(
    leaf: PrimitiveLeaf<'_>,
    field: &FieldIR,
    config: &super::MacroConfig,
) -> TokenStream {
    super::type_registry::full_dtype(leaf, &field.wrapper_shape, &config.external_paths)
}

/// Build the columnar emit pieces for one field. Routes every primitive
/// shape through the encoder IR, and every nested-struct/generic field
/// through the encoder's nested path (which covers every wrapper stack).
pub fn build_field_emit(
    field: &FieldIR,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
) -> FieldEmit {
    match field.leaf_spec.route() {
        LeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            build_nested_emit(field, config, idx, &type_path)
        }
        LeafRoute::Primitive(leaf) => build_primitive_emit(field, config, idx, it_ident, leaf),
        LeafRoute::Tuple(elements) => {
            super::encoder::build_tuple_field_emit(field, config, idx, elements)
        }
    }
}

fn build_nested_emit(
    field: &FieldIR,
    config: &super::MacroConfig,
    idx: usize,
    type_path: &TokenStream,
) -> FieldEmit {
    // The nested encoder paths run their own `for __df_derive_it in items`
    // loops to build their flat ref vec, so the access expression is
    // hard-rooted at the centralized populator-iter ident regardless of the
    // call site's outer-loop binding.
    let inner_it = idents::populator_iter();
    let access = it_access(field, &inner_it);
    let name = super::helpers::column_name_for_ident(&field.name);
    let ctx = NestedLeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name: &name,
        },
        ty: type_path,
        columnar_trait: &config.columnar_trait_path,
        to_df_trait: &config.to_dataframe_trait_path,
        paths: &config.external_paths,
    };
    let columnar = encoder::build_nested_encoder(&field.wrapper_shape, &ctx);
    FieldEmit::WholeColumn {
        builders: vec![columnar],
    }
}

/// Build the columnar emit pieces for a primitive-routed field. `[Vec, ...]`
/// shapes produce `Encoder::Multi` (the encoder packs precount, buffers,
/// fill loop, leaf array, list stacking, and the rename + push into one
/// self-contained block). Bare and `[Option]` shapes produce `Encoder::Leaf`
/// with decls + push + finisher split across the three slots.
fn build_primitive_emit(
    field: &FieldIR,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
    leaf: PrimitiveLeaf<'_>,
) -> FieldEmit {
    let name = super::helpers::column_name_for_ident(&field.name);
    let access = it_access(field, it_ident);
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name: &name,
        },
        decimal128_encode_trait: &config.decimal128_encode_trait_path,
        paths: &config.external_paths,
    };
    let enc = encoder::build_encoder(leaf, &field.wrapper_shape, &leaf_ctx);
    match enc {
        Encoder::Leaf {
            decls,
            push,
            series,
        } => {
            let builder = quote! {{
                let s = #series;
                columns.push(s.into());
            }};
            FieldEmit::RowWise {
                decls,
                push,
                builders: vec![builder],
            }
        }
        Encoder::Multi { columnar } => FieldEmit::WholeColumn {
            builders: vec![columnar],
        },
    }
}
