//! Per-field codegen entry point. Translates a [`FieldIR`] into the four
//! pieces of generated code each field contributes to: schema entries,
//! empty-series rows, columnar populator decls/pushes/finishes.
//!
//! The columnar path routes through the encoder IR in
//! [`super::encoder`] for every primitive shape — bare leaves, arbitrary
//! `Option<…<Option<T>>>` stacks, and every vec-bearing wrapper stack.

use crate::ir::{BaseType, FieldIR, PrimitiveTransform, vec_count};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::encoder::{self, BaseCtx, Encoder, LeafCtx, NestedLeafCtx, build_type_path, idents};

/// Per-field columnar emit pieces. The columnar pipeline concatenates
/// every field's `decls` ahead of the per-row push loop, splices every
/// field's `push` inside the loop, and concatenates every field's
/// `builders` after the loop to assemble the final `columns: Vec<Column>`.
pub struct FieldEmit {
    pub decls: Vec<TokenStream>,
    pub push: TokenStream,
    pub builders: Vec<TokenStream>,
}

/// Routing decision for one field. `Primitive` covers every base type that
/// resolves to a single Polars column at the encoder boundary (numerics,
/// strings, decimals, dates, plus stringy-transformed structs/generics).
/// `Nested` covers concrete structs and generic type parameters that
/// resolve to one Polars column per inner schema entry — `type_path` is
/// the splicable `Foo::<M>` / `T` token stream and `list_layers` is the
/// number of `Vec<…>` wrappers around the nested type.
enum FieldRoute {
    Primitive,
    Nested {
        type_path: TokenStream,
        list_layers: usize,
    },
}

/// Whether a transform routes the field through the primitive path even
/// when the base type is a struct/generic. `to_string`/`as_str` over a
/// nested type produce a `String` column, not a nested struct column.
const fn is_stringy(t: Option<&PrimitiveTransform>) -> bool {
    matches!(
        t,
        Some(PrimitiveTransform::ToString | PrimitiveTransform::AsStr)
    )
}

fn classify_field(field: &FieldIR) -> FieldRoute {
    if is_stringy(field.transform.as_ref()) {
        return FieldRoute::Primitive;
    }
    match &field.base_type {
        BaseType::Struct(id, args) => FieldRoute::Nested {
            type_path: build_type_path(id, args.as_ref()),
            list_layers: vec_count(&field.wrappers),
        },
        BaseType::Generic(id) => FieldRoute::Nested {
            type_path: quote! { #id },
            list_layers: vec_count(&field.wrappers),
        },
        BaseType::F64
        | BaseType::F32
        | BaseType::I64
        | BaseType::U64
        | BaseType::I32
        | BaseType::U32
        | BaseType::I16
        | BaseType::U16
        | BaseType::I8
        | BaseType::U8
        | BaseType::Bool
        | BaseType::String
        | BaseType::ISize
        | BaseType::USize
        | BaseType::DateTimeUtc
        | BaseType::Decimal => FieldRoute::Primitive,
    }
}

/// `<it>.<field>` — the field-access expression rooted at a per-row
/// iterator binding. Used by every emit path that needs to reach into
/// the per-row item.
fn it_access(field: &FieldIR, it_ident: &Ident) -> TokenStream {
    field.field_index.map_or_else(
        || {
            let id = &field.name;
            quote! { #it_ident.#id }
        },
        |i| {
            let li = syn::Index::from(i);
            quote! { #it_ident.#li }
        },
    )
}

/// Build the schema entries token expression for one field. Evaluates to a
/// `Vec<(String, DataType)>` at runtime — primitive fields return a
/// one-element vec, nested fields return one entry per inner schema column
/// (with the parent name prefixed).
pub fn build_schema_entries(field: &FieldIR) -> TokenStream {
    let name = field.name.to_string();
    match classify_field(field) {
        FieldRoute::Nested {
            type_path,
            list_layers,
        } => super::nested::generate_schema_entries_for_struct(&type_path, &name, list_layers),
        FieldRoute::Primitive => {
            let dtype = field_full_dtype(field);
            quote! { ::std::vec![(::std::string::String::from(#name), #dtype)] }
        }
    }
}

/// Build the empty-series token expression for one field. Evaluates to a
/// `Vec<Column>` at runtime — primitive fields produce one empty Series,
/// nested fields produce one empty Series per inner schema column.
pub fn build_empty_series(field: &FieldIR) -> TokenStream {
    let name = field.name.to_string();
    match classify_field(field) {
        FieldRoute::Nested { type_path, .. } => {
            super::nested::nested_empty_series_row(&type_path, &name, &field.wrappers)
        }
        FieldRoute::Primitive => {
            let dtype = field_full_dtype(field);
            let pp = super::polars_paths::prelude();
            quote! { ::std::vec![#pp::Series::new_empty(#name.into(), &#dtype).into()] }
        }
    }
}

fn field_full_dtype(field: &FieldIR) -> TokenStream {
    super::type_registry::compute_full_dtype(
        &field.base_type,
        field.transform.as_ref(),
        &field.wrappers,
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
    match classify_field(field) {
        FieldRoute::Nested { type_path, .. } => {
            let pa_root = super::polars_paths::polars_arrow_root();
            build_nested_emit(field, config, idx, &type_path, &pa_root)
        }
        FieldRoute::Primitive => build_primitive_emit(field, config, idx, it_ident),
    }
}

fn build_nested_emit(
    field: &FieldIR,
    config: &super::MacroConfig,
    idx: usize,
    type_path: &TokenStream,
    pa_root: &TokenStream,
) -> FieldEmit {
    // The nested encoder paths run their own `iter().map(|__df_derive_it| ...)`
    // loops to build their flat ref vec, so the access expression is
    // hard-rooted at the centralized populator-iter ident regardless of the
    // call site's outer-loop binding.
    let inner_it = idents::populator_iter();
    let access = it_access(field, &inner_it);
    let name = field.name.to_string();
    let ctx = NestedLeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name: &name,
        },
        ty: type_path,
        columnar_trait: &config.columnar_trait_path,
        to_df_trait: &config.to_dataframe_trait_path,
        pa_root,
    };
    let enc = encoder::build_nested_encoder(&field.wrappers, &ctx);
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
    let name = field.name.to_string();
    let access = it_access(field, it_ident);
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &access,
            idx,
            name: &name,
        },
        decimal128_encode_trait: &config.decimal128_encode_trait_path,
    };
    let enc = encoder::build_encoder(
        &field.base_type,
        field.transform.as_ref(),
        &field.wrappers,
        &leaf_ctx,
    );
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
