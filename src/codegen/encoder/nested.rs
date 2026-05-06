//! Nested-struct/generic encoder paths (`CollectThenBulk` leaves).
//!
//! Ports the seven nested-struct/generic shapes (`[]`, `[Option]`,
//! `[Vec]`, `[Option, Vec]`, `[Vec, Option]`, `[Option, Vec, Option]`,
//! `[Vec, Vec]`) into the encoder IR. Each shape is built up from a single
//! `CollectThenBulk` leaf (which knows how to call
//! `<T as Columnar>::columnar_from_refs(&refs)`) plus the wrapper-stack-shaped
//! gather/scatter machinery in this module.
//!
//! The invariant: every `LargeListArray::new` routes through the in-scope free
//! helper `__df_derive_assemble_list_series_unchecked` (defined at the top of
//! each derive's `const _: () = { ... };` scope), keeping `unsafe` out of any
//! `Self`-bearing impl method so `clippy::unsafe_derive_deserialize` stays
//! silent on downstream `#[derive(ToDataFrame, Deserialize)]` types.
//!
//! Every shape produces an `Encoder::Multi { columnar }` because the inner
//! `DataFrame` carries one column per inner schema entry of `T`. The block
//! pushes one Series per inner schema column onto the call site's `columns`
//! vec, with the parent name prefixed onto each inner column name.

use crate::ir::{VecLayers, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;

use super::emit::{nested_consume_columns, vec_emit_general};
use super::idents;
use super::leaf_kind::{CollectThenBulk, LeafKind};
use super::{BaseCtx, Encoder, collapse_options_to_ref};

/// Per-call-site context for nested-struct/generic encoders. Carries the
/// `polars-arrow` crate root (so the combinators don't re-resolve it per
/// call) plus the type-as-path expression and the fully-qualified trait
/// paths used in UFCS calls (`<#ty as #columnar_trait>::columnar_from_refs`,
/// `<#ty as #to_df_trait>::schema`).
pub struct NestedLeafCtx<'a> {
    pub base: BaseCtx<'a>,
    pub ty: &'a TokenStream,
    pub columnar_trait: &'a TokenStream,
    pub to_df_trait: &'a TokenStream,
    pub pa_root: &'a TokenStream,
}

/// Per-shape identifier bundle for the bare-leaf and option-leaf nested
/// encoders. The depth-N (`Vec`-bearing) path no longer threads through
/// here — it routes through the unified emitter via [`CollectThenBulk`].
struct NestedIdents {
    /// `Vec<&T>` flat ref accumulator.
    flat: syn::Ident,
    /// `Vec<Option<IdxSize>>` per-element positions for the inner-Option
    /// scatter case (option-leaf encoder only).
    positions: syn::Ident,
    /// Inner `DataFrame` returned by `columnar_from_refs`.
    df: syn::Ident,
    /// `IdxCa` built from `positions` for the option-leaf scatter case.
    take: syn::Ident,
}

impl NestedIdents {
    fn new(idx: usize) -> Self {
        Self {
            flat: idents::nested_flat(idx),
            positions: idents::nested_positions(idx),
            df: idents::nested_df(idx),
            take: idents::nested_take(idx),
        }
    }
}

/// Build the bare-leaf nested encoder (`payload: T`). Gathers refs into
/// `Vec<&T>`, calls `columnar_from_refs` once, and per inner schema column
/// pulls the materialized `Series` straight out of the resulting `DataFrame`
/// (no list-array wrapping; the parent column is the inner column).
fn nested_leaf_encoder(ctx: &NestedLeafCtx<'_>) -> Encoder {
    let NestedLeafCtx {
        base: BaseCtx { access, idx, name },
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let df = &ids.df;
    let col_name = idents::nested_col_name();
    let inner_expr = quote! {
        #df.column(#col_name)?
            .as_materialized_series()
            .clone()
    };
    let columnar = nested_consume_columns(name, to_df_trait, ty, &inner_expr);
    let it = idents::populator_iter();
    let setup = quote! {
        let #flat: ::std::vec::Vec<&#ty> = items
            .iter()
            .map(|#it| &(#access))
            .collect();
        let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
    };
    let columnar_block = quote! {{ #setup #columnar }};
    Encoder::Multi {
        columnar: columnar_block,
    }
}

/// `option(nested_leaf)` — `[Option]` (or any consecutive run of `Option`s
/// over a struct/generic, since Polars folds nested Nones into one validity
/// bit). For `option_layers >= 2`, the caller pre-collapses the access into
/// an `Option<&T>` value-expression; the scan then reads the value directly
/// without a `&`.
///
/// Splits each row's `Option<T>` into a flat ref slice plus a
/// `Vec<Option<IdxSize>>` of positions. Three runtime branches:
/// - all None: emit one typed-null Series of length `items.len()` per inner
///   schema column.
/// - all Some (no scatter needed): pull each column straight from the inner
///   `DataFrame`, no `take`.
/// - mixed: build an `IdxCa` over positions and `take` per inner column to
///   scatter values back over the original row positions.
fn nested_option_encoder_collapsed(ctx: &NestedLeafCtx<'_>, option_layers: usize) -> Encoder {
    // For `option_layers >= 2`, `#access` is an `as_ref().and_then(...)`
    // chain returning `Option<&T>` directly — we match it by value.
    // For `option_layers == 1`, `#access` is the raw `&Option<T>` field
    // expression — we match by reference. The two arms produce slightly
    // different scans because the bound `__df_derive_v` is `&T` either way,
    // but the surrounding match expression differs.
    let access_ts = ctx.base.access.clone();
    let match_expr = if option_layers >= 2 {
        quote! { (#access_ts) }
    } else {
        quote! { &(#access_ts) }
    };
    nested_option_encoder_impl(ctx, &match_expr)
}

fn nested_option_encoder_impl(ctx: &NestedLeafCtx<'_>, match_expr: &TokenStream) -> Encoder {
    let NestedLeafCtx {
        base: BaseCtx {
            access: _,
            idx,
            name,
        },
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let pp = crate::codegen::polars_paths::prelude();
    let ids = NestedIdents::new(idx);
    let flat = &ids.flat;
    let positions = &ids.positions;
    let df = &ids.df;
    let take = &ids.take;
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();

    let direct_inner = quote! {
        #df.column(#col_name)?
            .as_materialized_series()
            .clone()
    };
    let take_inner = quote! {{
        let #inner_full = #df
            .column(#col_name)?
            .as_materialized_series();
        #inner_full.take(&#take)?
    }};
    let null_inner = quote! {
        #pp::Series::new_empty("".into(), #dtype)
            .extend_constant(#pp::AnyValue::Null, items.len())?
    };

    let it = idents::populator_iter();
    let v = idents::leaf_value();
    let scan = quote! {
        let mut #flat: ::std::vec::Vec<&#ty> = ::std::vec::Vec::with_capacity(items.len());
        let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for #it in items {
            match #match_expr {
                ::std::option::Option::Some(#v) => {
                    #positions.push(::std::option::Option::Some(
                        #flat.len() as #pp::IdxSize,
                    ));
                    #flat.push(#v);
                }
                ::std::option::Option::None => {
                    #positions.push(::std::option::Option::None);
                }
            }
        }
    };
    let consume_direct = nested_consume_columns(name, to_df_trait, ty, &direct_inner);
    let consume_take = nested_consume_columns(name, to_df_trait, ty, &take_inner);
    let consume_null = nested_consume_columns(name, to_df_trait, ty, &null_inner);
    let columnar_block = quote! {{
        #scan
        if #flat.is_empty() {
            #consume_null
        } else if #flat.len() == items.len() {
            let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
            #consume_direct
        } else {
            let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
            let #take: #pp::IdxCa =
                <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #positions.iter().copied(),
                );
            #consume_take
        }
    }};
    Encoder::Multi {
        columnar: columnar_block,
    }
}

// --- Generalized depth-N nested encoder ---

/// Build the depth-N nested vec encoder for an arbitrary [`VecLayers`].
/// Handles per-layer outer-Option (validity bitmap), inner-Option
/// (per-element positions + scatter via `IdxCa::take`), and any mix
/// thereof. Replaces the seven hand-written shape variants.
///
/// Delegates the depth-N scaffolding (precount, storage, scan, assemble)
/// to the unified [`vec_emit_general`]; constructs a
/// [`LeafKind::CollectThenBulk`] payload from the field's type/trait
/// plumbing.
fn nested_vec_encoder_general(ctx: &NestedLeafCtx<'_>, shape: &VecLayers) -> Encoder {
    let NestedLeafCtx {
        base: BaseCtx { access, idx, name },
        ty,
        columnar_trait,
        to_df_trait,
        pa_root: _,
    } = *ctx;
    let ctb = CollectThenBulk {
        ty,
        columnar_trait,
        to_df_trait,
        name,
        idx,
    };
    let kind = LeafKind::CollectThenBulk(ctb);
    let columnar = vec_emit_general(&kind, access, idx, shape);
    Encoder::Multi { columnar }
}

/// Top-level dispatcher for the nested-struct/generic encoder paths.
/// After Step 4 this covers every wrapper stack the parser accepts —
/// the `[]` and `[Option]` shapes use dedicated leaf encoders; every
/// `Vec`-bearing shape (including deep nestings, mid-stack `Option`s,
/// outer-list validity) routes through the depth-N general encoder.
pub fn build_nested_encoder(wrapper: &WrapperShape, ctx: &NestedLeafCtx<'_>) -> Encoder {
    match wrapper {
        WrapperShape::Leaf { option_layers: 0 } => nested_leaf_encoder(ctx),
        WrapperShape::Leaf {
            option_layers: layers,
        } => {
            let layers = *layers;
            // Collapse N consecutive Options into a single `Option<&T>`
            // before invoking the option-leaf encoder. Polars folds every
            // nested None into one validity bit, so `Some(None)` and
            // outer `None` produce the same `AnyValue::Null`. The
            // intermediate access expression is `(...).as_ref().and_then(...)`
            // which evaluates to `Option<&T>` and matches the option-leaf
            // encoder's expected access type for the single-Option case.
            let collapsed_access = if layers >= 2 {
                let chain = collapse_options_to_ref(ctx.base.access, layers);
                quote! { (#chain) }
            } else {
                ctx.base.access.clone()
            };
            let new_ctx = NestedLeafCtx {
                base: BaseCtx {
                    access: &collapsed_access,
                    idx: ctx.base.idx,
                    name: ctx.base.name,
                },
                ty: ctx.ty,
                columnar_trait: ctx.columnar_trait,
                to_df_trait: ctx.to_df_trait,
                pa_root: ctx.pa_root,
            };
            nested_option_encoder_collapsed(&new_ctx, layers)
        }
        WrapperShape::Vec(shape) => nested_vec_encoder_general(ctx, shape),
    }
}
