//! Per-field codegen entry point. Translates a [`FieldIR`] into the four
//! pieces of generated code each field contributes to: schema entries,
//! empty-series rows, columnar populator decls/pushes/finishes.
//!
//! The columnar path routes through the encoder IR in
//! [`super::encoder`] for every primitive shape — bare leaves, arbitrary
//! `Option<…<Option<T>>>` stacks, and every vec-bearing wrapper stack.

use crate::ir::{FieldIR, LeafSpec};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::encoder::{self, BaseCtx, Encoder, LeafCtx, NestedLeafCtx, idents, struct_type_tokens};

/// Per-field columnar emit pieces. The columnar pipeline concatenates
/// every field's `decls` ahead of the per-row push loop, splices every
/// field's `push` inside the loop, and concatenates every field's
/// `builders` after the loop to assemble the final `columns: Vec<Column>`.
pub(in crate::codegen) struct FieldEmit {
    pub decls: Vec<TokenStream>,
    pub push: TokenStream,
    pub builders: Vec<TokenStream>,
}

/// Routing decision for one field. `Primitive` covers every leaf that
/// resolves to a single Polars column at the encoder boundary (numerics,
/// strings, decimals, dates, plus stringy-transformed structs/generics).
/// `Nested` covers concrete structs and generic type parameters that
/// resolve to one Polars column per inner schema entry — `type_path` is
/// the splicable `Foo::<M>` / `T` token stream. `Tuple` covers tuple-typed
/// fields where each element contributes its own column(s); the parent
/// wrapper stack distributes across every element column at codegen time.
pub(in crate::codegen) enum FieldRoute {
    Primitive,
    Nested { type_path: TokenStream },
    Tuple,
}

impl LeafSpec {
    /// Single source of truth for primitive-vs-nested routing.
    /// `Struct`/`Generic` without a stringy override (`as_str`/`as_string`)
    /// goes to the nested encoder; everything else (including `as_str` /
    /// `as_string` over a struct or generic — which produces a `String`
    /// column, not a nested struct column) stays primitive. The parser's
    /// legality matrix folds the stringy overrides into
    /// `LeafSpec::AsStr`/`LeafSpec::AsString`, so the destructure here is
    /// structurally exhaustive — no `is_stringy` bridge.
    fn route(&self) -> FieldRoute {
        match self {
            Self::Struct(ty) => FieldRoute::Nested {
                type_path: struct_type_tokens(ty),
            },
            Self::Generic(id) => FieldRoute::Nested {
                type_path: quote! { #id },
            },
            Self::Tuple(_) => FieldRoute::Tuple,
            Self::Numeric(_)
            | Self::String
            | Self::Bool
            | Self::DateTime(_)
            | Self::NaiveDateTime(_)
            | Self::NaiveDate
            | Self::NaiveTime
            | Self::Duration { .. }
            | Self::Decimal { .. }
            | Self::AsString(_)
            | Self::AsStr(_)
            | Self::Binary => FieldRoute::Primitive,
        }
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
/// share the Primitive-vs-Nested classification; only the leaf token shape
/// (and the runtime accumulator inside [`super::nested`]) varies.
#[derive(Clone, Copy)]
pub(in crate::codegen) enum EmitMode {
    SchemaEntries,
    EmptyRows,
}

/// Shared per-field emitter for the schema / empty-rows pair. Classifies
/// the field once, then dispatches to the matching [`super::nested`]
/// runtime helper for nested fields, or emits a one-element vec literal
/// for primitive fields. The two leaf shapes are byte-equivalent to the
/// pre-refactor `build_schema_entries` / `build_empty_series` emissions.
fn build_field_entries(
    field: &FieldIR,
    mode: EmitMode,
    config: &super::MacroConfig,
) -> TokenStream {
    let name = super::helpers::column_name_for_ident(&field.name);
    match (field.leaf_spec.route(), mode) {
        (FieldRoute::Nested { type_path }, EmitMode::SchemaEntries) => {
            super::nested::generate_schema_entries_for_struct(
                &type_path,
                &config.to_dataframe_trait_path,
                &name,
                field.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        (FieldRoute::Nested { type_path }, EmitMode::EmptyRows) => {
            super::nested::nested_empty_series_row(
                &type_path,
                &config.to_dataframe_trait_path,
                &name,
                field.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        (FieldRoute::Primitive, EmitMode::SchemaEntries) => {
            let dtype = field_full_dtype(field, config);
            quote! { ::std::vec![(::std::string::String::from(#name), #dtype)] }
        }
        (FieldRoute::Primitive, EmitMode::EmptyRows) => {
            let dtype = field_full_dtype(field, config);
            let pp = config.external_paths.prelude();
            quote! { ::std::vec![#pp::Series::new_empty(#name.into(), &#dtype).into()] }
        }
        (FieldRoute::Tuple, _) => {
            let LeafSpec::Tuple(elements) = &field.leaf_spec else {
                unreachable!("FieldRoute::Tuple implies LeafSpec::Tuple")
            };
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

fn field_full_dtype(field: &FieldIR, config: &super::MacroConfig) -> TokenStream {
    super::type_registry::full_dtype(
        &field.leaf_spec,
        &field.wrapper_shape,
        &config.external_paths,
    )
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
        FieldRoute::Nested { type_path } => build_nested_emit(field, config, idx, &type_path),
        FieldRoute::Primitive => build_primitive_emit(field, config, idx, it_ident),
        FieldRoute::Tuple => {
            let LeafSpec::Tuple(elements) = &field.leaf_spec else {
                unreachable!("FieldRoute::Tuple implies LeafSpec::Tuple")
            };
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
    let enc = encoder::build_nested_encoder(&field.wrapper_shape, &ctx);
    let Encoder::Multi { columnar } = enc else {
        unreachable!("nested encoder must produce a multi-column encoder")
    };
    FieldEmit {
        decls: Vec::new(),
        push: TokenStream::new(),
        builders: vec![columnar],
    }
}

/// Build the columnar emit pieces for a primitive-routed field. `[Vec, ...]`
/// shapes produce `Encoder::Multi` (the encoder packs precount, buffers,
/// fill loop, leaf array, list stacking, and the rename + push into one
/// self-contained block placed AFTER the per-row loop — matches the legacy
/// direct-fast-path emitters' cache locality, see `vec_vec_i32` /
/// `vec_opt_datetime` benches: emitting in decls regresses ~4% relative to
/// placing the same work post-loop). Bare and `[Option]` shapes produce
/// `Encoder::Leaf` with decls + push + finisher splayed across the three
/// slots.
fn build_primitive_emit(
    field: &FieldIR,
    config: &super::MacroConfig,
    idx: usize,
    it_ident: &Ident,
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
    let enc = encoder::build_encoder(&field.leaf_spec, &field.wrapper_shape, &leaf_ctx);
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
            FieldEmit {
                decls,
                push,
                builders: vec![builder],
            }
        }
        Encoder::Multi { columnar } => FieldEmit {
            decls: Vec::new(),
            push: TokenStream::new(),
            builders: vec![columnar],
        },
    }
}
