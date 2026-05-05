//! Vec combinator + primitive vec dispatch + `build_leaf`.
//!
//! Implements the `vec(inner)` combinator that fuses N consecutive `Vec`
//! layers into a single bulk emission (one flat values buffer at the
//! deepest layer, one optional inner validity bitmap, one pair of offsets
//! per layer, one optional outer-list validity bitmap per layer that has
//! an adjoining `Option`). The bulk-fusion invariant lives here.
//!
//! Also hosts `build_leaf` — the dispatcher from `(base, transform)` to a
//! primitive leaf encoder — because it sits at the leaf/vec boundary and
//! `try_build_vec_encoder` shares its (base, transform) coverage matrix.

use crate::ir::{DateTimeUnit, PrimitiveTransform};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents;
use super::leaf::{LeafSpec, validity_into_option};
use super::shape_walk::{
    LayerIdents, LayerWrap, OwnPolicy, ShapePrecount, ShapeScan, shape_assemble_list_stack,
    shape_offsets_decls, shape_validity_decls,
};
use super::{Encoder, LeafCtx, LeafShape, StringyBase, VecShape, collapse_options_to_ref, leaf};

// --- Vec combinator ---

/// Per-leaf-kind plumbing the `vec_for_*` combinator needs to assemble a
/// flat `Vec<Native>` / `MutableBinaryViewArray<str>` / `MutableBitmap`
/// over all elements across all outer rows, then wrap it in one or two
/// `LargeListArray::new` layers. Each variant pairs a per-element store
/// expression (used by both bare and option-Some arms) with the dtype
/// tokens needed to build the leaf array and route through the in-scope
/// `__df_derive_assemble_list_series_unchecked` helper.
///
/// Per-leaf-kind classification used by the `vec(inner)` combinator. The
/// per-element push and the leaf-array build differ across these variants;
/// every other piece of the bulk emit (precount loops, offsets vecs, list
/// stacking, `__df_derive_assemble_list_series_unchecked` routing) is shared.
///
/// `Bool` carries no extra fields because the bit-packed values bitmap and
/// the no-validity / pre-filled-validity arrays are constructed the same way
/// regardless of context — the only knob is `has_inner_option`. Bare bool
/// (`Vec<bool>` / `Vec<Vec<bool>>`) follows a different layout entirely
/// (flat `Vec<bool>` for depth 1, bit-packed bitmap for depth 2) and is
/// served by a dedicated builder, not this enum.
enum VecLeafSpec {
    /// Numeric (i8/i16/.../f64) and the single-`i128`/`i64` transform leaves
    /// (`Decimal` -> `i128`, `DateTime` -> `i64`). All share
    /// `PrimitiveArray::new` over `Vec<Native>` plus an optional
    /// inner-validity bitmap.
    ///
    /// `value_expr` materializes the leaf value from the binding
    /// `__df_derive_v` — the loop binding is `&T` for bare shapes and the
    /// `Some(v)` arm of the inner option-match for nullable shapes (same
    /// binding name in both, so a single expression suffices).
    Numeric {
        native: TokenStream,
        value_expr: TokenStream,
        /// Adds an anonymous `use #decimal128_encode_trait as _;` so the
        /// dot-syntax `try_to_i128_mantissa` resolves on `&Decimal`.
        needs_decimal_import: bool,
    },
    /// `String` / `to_string` / `as_str` over `MutableBinaryViewArray<str>`.
    /// `value_expr` materializes a `&str` from the `__df_derive_v` binding
    /// (same convention as `Numeric`). `extra_decls` covers the per-row
    /// `String` scratch used by `to_string` (placed before the MBVA so
    /// cleanup ordering matches prior emission).
    StringLike {
        value_expr: TokenStream,
        extra_decls: Vec<TokenStream>,
    },
    /// Inner-Option bool — bit-packed values bitmap (pre-filled `false`)
    /// plus a parallel validity bitmap (pre-filled `true`). Bare bool is
    /// served separately because it gets a flat `Vec<bool>` at depth 1.
    Bool,
    /// Bare bool — bit-packed values bitmap (pre-filled `false`), no
    /// inner-Option. Used for depth >= 2 (and any shape that has an outer
    /// validity layer above the bool leaf, since the depth-1 fast path
    /// requires both no inner option and no outer option).
    BoolBare,
}

/// Bool-specific helper: returns the leaf-array construction tokens given
/// whether the parent shape carries an inner validity bitmap.
fn bool_leaf_array_tokens(
    pa_root: &TokenStream,
    has_inner_option: bool,
    values_ident: &syn::Ident,
    validity_ident: &syn::Ident,
) -> TokenStream {
    if has_inner_option {
        let valid_opt = validity_into_option(validity_ident);
        quote! {
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#values_ident),
                #valid_opt,
            )
        }
    } else {
        quote! {
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#values_ident),
                ::std::option::Option::None,
            )
        }
    }
}

/// Per-`Vec` layer ident set for the flat-vec path. Layer `i` is the
/// `i`-th `Vec` from the outside; layer `depth-1` is the innermost (its
/// `offsets` track flat-leaf counts; deeper layers' `offsets` track
/// child-list counts). The `validity_mb` field is allocated only when
/// `has_outer_validity` for that layer.
fn vec_layer_idents(depth: usize) -> Vec<LayerIdents> {
    (0..depth)
        .map(|i| LayerIdents {
            offsets: idents::vec_layer_offsets(i),
            offsets_buf: idents::vec_layer_offsets_buf(i),
            validity_mb: idents::vec_layer_validity(i),
            validity_bm: idents::vec_layer_validity_bm(i),
            bind: idents::vec_layer_bind(i),
        })
        .collect()
}

/// Build the entire `vec(inner)` (or deeper) emit block for a normalized
/// [`VecShape`]. Emits a single `let __df_derive_field_series_<idx> = { ... };`
/// declaration the caller splices into the populator's pre-loop decls.
///
/// The bulk-fusion contract: regardless of depth, the leaf storage (flat
/// values buffer + optional validity bitmap) is allocated and populated
/// once; the layer stack adds one `LargeListArray::new` per `Vec` wrapper
/// and one `MutableBitmap` per layer that has an outer-`Option`.
fn vec_emit_decl(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecShape,
    leaf_dtype_tokens: &TokenStream,
) -> TokenStream {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    let pp = crate::codegen::polars_paths::prelude();
    let access = ctx.access;
    let series_local = vec_encoder_series_local(ctx.idx);
    let leaf_bind = idents::leaf_value();
    let layers = vec_layer_idents(shape.depth());

    let (precount_decls, leaf_capacity_expr) = vec_precount_pieces(access, shape, &layers);
    let (leaf_storage_decls, per_elem_push, leaf_arr_expr) = build_vec_leaf_pieces(
        spec,
        shape.has_inner_option(),
        &leaf_capacity_expr,
        &pa_root,
    );
    // Decimal mantissa rescale dispatches through the `Decimal128Encode`
    // trait via dot syntax — the trait must be in scope so method
    // resolution finds it. Anonymous `use ... as _;` keeps the user's
    // namespace clean. Other `Numeric` variants and `StringLike` / `Bool`
    // don't reference any user trait.
    let extra_imports = if let VecLeafSpec::Numeric {
        needs_decimal_import: true,
        ..
    } = spec
    {
        let trait_path = ctx.decimal128_encode_trait;
        quote! { use #trait_path as _; }
    } else {
        TokenStream::new()
    };

    let leaf_offsets_post_push = leaf_offsets_post_push_tokens(spec);
    let offsets_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.offsets).collect();
    let validity_idents: Vec<&syn::Ident> = layers.iter().map(|l| &l.validity_mb).collect();
    let counter_for_depth = |i: usize| idents::vec_layer_total_token(i);
    let offsets_decls = shape_offsets_decls(&offsets_idents, &counter_for_depth);
    let validity_decls =
        shape_validity_decls(shape, &validity_idents, &counter_for_depth, &pa_root);
    let push_loops = build_vec_push_loops(
        access,
        shape,
        &layers,
        &leaf_bind,
        &per_elem_push,
        &leaf_offsets_post_push,
    );
    let final_assemble = vec_final_assemble(shape, &layers, leaf_dtype_tokens, &pa_root, &pp);

    quote! {
        let #series_local: #pp::Series = {
            #extra_imports
            #precount_decls
            #leaf_storage_decls
            #offsets_decls
            #validity_decls
            #push_loops
            #leaf_arr_expr
            #final_assemble
        };
    }
}

/// Precount loop + leaf-capacity expression for the depth-N bulk-vec emit.
/// Returns `(decls, leaf_capacity)` — the leaf capacity is the running total
/// of leaf elements summed across every nested `Vec` layer of every outer row.
///
/// Routes through the shared [`ShapePrecount`] walker; the per-row body
/// mirrors `build_vec_push_loops` so precount and scan walk the same
/// `Some/None` arms in lock-step. Layers with `has_outer_validity` skip both
/// the layer-counter increment and the recursion on `None`, matching the
/// runtime push logic that records a repeat-offset for the null cell.
fn vec_precount_pieces(
    access: &TokenStream,
    shape: &VecShape,
    layers: &[LayerIdents],
) -> (TokenStream, TokenStream) {
    let depth = shape.depth();
    let total_leaves = idents::total_leaves();
    let layer_counters: Vec<syn::Ident> = (0..depth.saturating_sub(1))
        .map(idents::vec_layer_total)
        .collect();
    let pre = ShapePrecount {
        shape,
        access,
        layers,
        outer_some_prefix: idents::VEC_OUTER_SOME_PREFIX,
        total_counter: &total_leaves,
        layer_counters: &layer_counters,
    }
    .build();
    (pre, quote! { #total_leaves })
}

/// Build the nested for-loop push body for the depth-N vec emit. Routes
/// through the shared [`ShapeScan`] walker (in `shape_walk`); the per-leaf
/// loop body splices `per_elem_push` and handles `inner_option_layers > 1`
/// via `collapse_options_to_ref`.
fn build_vec_push_loops(
    access: &TokenStream,
    shape: &VecShape,
    layers: &[LayerIdents],
    leaf_bind: &syn::Ident,
    per_elem_push: &TokenStream,
    leaf_offsets_post_push: &TokenStream,
) -> TokenStream {
    // The deepest-layer for-loop. The per_elem_push body expects
    // `__df_derive_v` to be either:
    // - `&T` directly (no inner-Option), bound by the for-loop, or
    // - `Option<&T>` (inner-Option), with the per-elem push then matching it.
    // To support `inner_option_layers > 1`, we collapse the for-loop binding
    // through `as_ref().and_then` into a single `Option<&T>` before splicing
    // the push body. The bare for-loop binding is `__df_derive_v_raw` and
    // the collapsed one becomes `__df_derive_v`.
    let leaf_body = |vec_bind: &TokenStream| -> TokenStream {
        if shape.has_inner_option() {
            if shape.inner_option_layers == 1 {
                quote! {
                    for #leaf_bind in #vec_bind.iter() {
                        #per_elem_push
                    }
                }
            } else {
                let raw_bind = idents::leaf_value_raw();
                let collapsed =
                    collapse_options_to_ref(&quote! { #raw_bind }, shape.inner_option_layers);
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind: ::std::option::Option<_> = #collapsed;
                        #per_elem_push
                    }
                }
            }
        } else {
            quote! {
                for #leaf_bind in #vec_bind.iter() {
                    #per_elem_push
                }
            }
        }
    };
    ShapeScan {
        shape,
        access,
        layers,
        outer_some_prefix: idents::VEC_OUTER_SOME_PREFIX,
        leaf_body: &leaf_body,
        leaf_offsets_post_push,
    }
    .build()
}

/// Stack `depth` `LargeListArray::new` layers and route the outermost one
/// through `__df_derive_assemble_list_series_unchecked` via the shared
/// [`shape_assemble_list_stack`] helper.
///
/// Builds a per-layer `freeze_decl` that the helper splices in just before
/// each layer's wrap (the freeze and the wrap stay co-located the way the
/// pre-refactor emission did — hoisting the freezes outside the wrap loop
/// is semantically identical but reproducibly regresses depth-N benches by
/// 4-12%). The leaf's arrow dtype is captured to a named local before the
/// leaf is boxed into an `ArrayRef` so the dtype access keeps its static
/// dispatch shape.
fn vec_final_assemble(
    shape: &VecShape,
    layers: &[LayerIdents],
    leaf_dtype_tokens: &TokenStream,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let depth = shape.depth();
    let mut wrap_layers: Vec<LayerWrap<'_>> = Vec::with_capacity(depth);
    for (cur, layer) in layers.iter().enumerate() {
        let offsets = &layer.offsets;
        let buf_id = &layer.offsets_buf;
        let mut freeze_decl = quote! {
            let #buf_id: #pa_root::offset::OffsetsBuffer<i64> =
                #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
        };
        let validity_bm = if shape.layers[cur].has_outer_validity() {
            let validity_mb = &layer.validity_mb;
            let bm_id = &layer.validity_bm;
            freeze_decl.extend(quote! {
                let #bm_id: #pa_root::bitmap::Bitmap =
                    <#pa_root::bitmap::Bitmap as ::core::convert::From<
                        #pa_root::bitmap::MutableBitmap,
                    >>::from(#validity_mb);
            });
            Some(bm_id)
        } else {
            None
        };
        // The flat-vec path freezes each per-layer offsets buffer once and
        // then uses it in exactly one `LargeListArray::new` (no shared
        // dispatch arms, no per-column iteration), so the helper can move
        // it into the wrap. The freeze itself is interleaved with the
        // wrap via [`LayerWrap::freeze_decl`]: hoisting all freezes above
        // the wrap loop reproducibly regresses depth-N benches by 4-12%
        // even though the resulting state is identical, so we replicate
        // the pre-refactor ordering layer-by-layer.
        wrap_layers.push(LayerWrap {
            offsets_buf: OwnPolicy::Move(buf_id),
            validity_bm,
            freeze_decl,
        });
    }
    let leaf_arr = idents::leaf_arr();
    // Capture the leaf's arrow dtype to a named local BEFORE boxing the
    // leaf — `Box::new(#leaf_arr) as ArrayRef` moves the typed leaf, so
    // a post-box `Array::dtype(&leaf)` would no longer compile, and a
    // post-box `Array::dtype(&seed)` would dispatch through the boxed
    // trait object's vtable (a virtual call that doesn't inline and
    // reproducibly regresses several depth-N benches by 5-12%).
    let seed_arrow_dtype_id = idents::seed_arrow_dtype();
    let seed_dtype_decl = quote! {
        let #seed_arrow_dtype_id: #pa_root::datatypes::ArrowDataType =
            #pa_root::array::Array::dtype(&#leaf_arr).clone();
    };
    let seed = quote! { ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef };
    let seed_dtype = quote! { #seed_arrow_dtype_id };
    let stack = shape_assemble_list_stack(
        seed,
        seed_dtype,
        &wrap_layers,
        leaf_dtype_tokens.clone(),
        &idents::vec_layer_list_arr,
    );
    quote! {
        #seed_dtype_decl
        #stack
    }
}

/// The expression that becomes `<offsets>.push(<expr> as i64)` at the
/// inner-vec-loop tail. Numeric / decimal / datetime use the flat-vec
/// length; the string-view path uses the MBVA `len()` (which is the count
/// of pushed views); the bool bit-packed-bitmap layout tracks a per-leaf
/// index so it can stay in sync with both the values and validity bitmaps.
fn leaf_offsets_post_push_tokens(spec: &VecLeafSpec) -> TokenStream {
    match spec {
        VecLeafSpec::Numeric { .. } => quote! { __df_derive_flat.len() },
        VecLeafSpec::StringLike { .. } => quote! { __df_derive_view_buf.len() },
        VecLeafSpec::Bool | VecLeafSpec::BoolBare => quote! { __df_derive_leaf_idx },
    }
}

/// Per-leaf storage decls + per-element push + leaf-array build.
///
/// Returns `(leaf_storage_decls, per_elem_push, leaf_arr_expr)`.
/// `leaf_storage_decls` covers the values buffer + (when `has_inner_option`)
/// the parallel `MutableBitmap` validity. `per_elem_push` is one per-element
/// `match` (option) or push (bare) referencing the binding `__df_derive_v`.
/// `leaf_arr_expr` produces the typed `PrimitiveArray<...>` /
/// `Utf8ViewArray` / `BooleanArray` value bound to `__df_derive_leaf_arr`.
fn build_vec_leaf_pieces(
    spec: &VecLeafSpec,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    match spec {
        VecLeafSpec::Numeric {
            native,
            value_expr,
            needs_decimal_import: _,
        } => numeric_leaf_pieces(
            native,
            value_expr,
            has_inner_option,
            leaf_capacity_expr,
            pa_root,
        ),
        VecLeafSpec::StringLike {
            value_expr,
            extra_decls,
        } => string_like_leaf_pieces(
            value_expr,
            extra_decls,
            has_inner_option,
            leaf_capacity_expr,
            pa_root,
        ),
        VecLeafSpec::Bool => {
            debug_assert!(
                has_inner_option,
                "VecLeafSpec::Bool only handles the inner-Option case",
            );
            bool_inner_option_leaf_pieces(leaf_capacity_expr, pa_root)
        }
        VecLeafSpec::BoolBare => {
            debug_assert!(
                !has_inner_option,
                "VecLeafSpec::BoolBare only handles the no-inner-Option case",
            );
            bool_bare_leaf_pieces(leaf_capacity_expr, pa_root)
        }
    }
}

/// Bit-packed values bitmap pre-filled `false`, with a per-leaf index
/// counter for `set(idx, true)`. Used by deeper-than-1 / outer-Option-bearing
/// `Vec<bool>` shapes that don't qualify for the depth-1 `from_slice`
/// fast path. The leaf-array build skips the validity argument.
fn bool_bare_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let storage = quote! {
        let mut __df_derive_values: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            __df_derive_b.extend_constant(#leaf_capacity_expr, false);
            __df_derive_b
        };
        let mut __df_derive_leaf_idx: usize = 0;
    };
    let push = quote! {
        if *__df_derive_v {
            __df_derive_values.set(__df_derive_leaf_idx, true);
        }
        __df_derive_leaf_idx += 1;
    };
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_arr_inner = bool_leaf_array_tokens(pa_root, false, &values_ident, &validity_ident);
    let leaf_arr = idents::leaf_arr();
    let leaf_arr_expr = quote! {
        let #leaf_arr: #pa_root::array::BooleanArray = #leaf_arr_inner;
    };
    (storage, push, leaf_arr_expr)
}

fn numeric_leaf_pieces(
    native: &TokenStream,
    value_expr: &TokenStream,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    // Numeric / Decimal / DateTime: `Vec<#native>` flat values.
    // Inner-Option carries a parallel `MutableBitmap` pre-filled `true`.
    let storage = if has_inner_option {
        quote! {
            let mut __df_derive_flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
            let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
                let mut __df_derive_b =
                    #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
                __df_derive_b.extend_constant(#leaf_capacity_expr, true);
                __df_derive_b
            };
        }
    } else {
        quote! {
            let mut __df_derive_flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
        }
    };
    // The bare and inner-Option arms both reference `__df_derive_v` — the
    // bare arm gets it from the loop binding directly (the for-loop in
    // `vec_emit_decl` binds the leaf as `__df_derive_v`), the option arm
    // gets it from the `Some(v)` pattern. Sharing one `value_expr` avoids
    // two near-duplicate per-spec expressions.
    // Push expressions match the legacy `try_gen_*` emitters' exact token
    // shape. Two distinct shapes survive in the legacy emitters:
    //
    // - `try_gen_vec_option_numeric_emit` (depth-1 `Vec<Option<numeric>>`):
    //   wraps the Some-arm value in `{ ... }`, e.g.
    //   `flat.push({ *__df_derive_v });`. Bench `08_complex_wrappers`
    //   reproducibly regresses ~5% when this wrap is dropped, even though
    //   rustc should see equivalent MIR.
    //
    // - `try_gen_nested_primitive_vec_emit` (depth-2+ `Vec<Vec<numeric>>`,
    //   no inner Option): no wrap; uses `flat.push(*v)` directly. Adding a
    //   wrap here regresses `vec_vec_i32` ~3-5%.
    //
    // The split is by `has_inner_option`: Some-arm gets the wrap, bare arm
    // does not. This locks in both bench-targeted shapes.
    let push = if has_inner_option {
        quote! {
            match __df_derive_v {
                ::std::option::Option::Some(__df_derive_v) => {
                    __df_derive_flat.push({ #value_expr });
                }
                ::std::option::Option::None => {
                    __df_derive_flat.push(<#native as ::std::default::Default>::default());
                    __df_derive_validity.set(__df_derive_flat.len() - 1, false);
                }
            }
        }
    } else {
        quote! {
            __df_derive_flat.push(#value_expr);
        }
    };
    let leaf_arr_expr = if has_inner_option {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::new(
                    <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                    __df_derive_flat.into(),
                    ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                        __df_derive_validity,
                    ),
                );
        }
    } else {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::from_vec(__df_derive_flat);
        }
    };
    (storage, push, leaf_arr_expr)
}

fn string_like_leaf_pieces(
    value_expr: &TokenStream,
    extra_decls: &[TokenStream],
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    // String / to_string / as_str: `MutableBinaryViewArray<str>` flat values.
    // Inner-Option uses a separate validity bitmap pre-filled `true`, plus
    // a row index that advances per element so `validity.set(i, false)`
    // hits the right bit on `None`.
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    for d in extra_decls {
        storage_parts.push(d.clone());
    }
    storage_parts.push(quote! {
        let mut __df_derive_view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        storage_parts.push(quote! {
            let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
                let mut __df_derive_b =
                    #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
                __df_derive_b.extend_constant(#leaf_capacity_expr, true);
                __df_derive_b
            };
            let mut __df_derive_leaf_idx: usize = 0;
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let push = if has_inner_option {
        quote! {
            match __df_derive_v {
                ::std::option::Option::Some(__df_derive_v) => {
                    __df_derive_view_buf.push_value_ignore_validity({ #value_expr });
                }
                ::std::option::Option::None => {
                    __df_derive_view_buf.push_value_ignore_validity("");
                    __df_derive_validity.set(__df_derive_leaf_idx, false);
                }
            }
            __df_derive_leaf_idx += 1;
        }
    } else {
        quote! {
            __df_derive_view_buf.push_value_ignore_validity({ #value_expr });
        }
    };
    let leaf_arr_expr = if has_inner_option {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::Utf8ViewArray = __df_derive_view_buf
                .freeze()
                .with_validity(
                    ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                        __df_derive_validity,
                    ),
                );
        }
    } else {
        quote! {
            let __df_derive_leaf_arr: #pa_root::array::Utf8ViewArray = __df_derive_view_buf.freeze();
        }
    };
    (storage, push, leaf_arr_expr)
}

fn bool_inner_option_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    // Bool with inner-Option only — bare bool is served by
    // `vec_encoder_bool_bare`. Bit-packed values bitmap (pre-filled `false`)
    // + parallel validity bitmap (pre-filled `true`) + leaf index counter so
    // `set(i, ...)` lands on the right bit.
    let storage = quote! {
        let mut __df_derive_values: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            __df_derive_b.extend_constant(#leaf_capacity_expr, false);
            __df_derive_b
        };
        let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            __df_derive_b.extend_constant(#leaf_capacity_expr, true);
            __df_derive_b
        };
        let mut __df_derive_leaf_idx: usize = 0;
    };
    let push = quote! {
        match __df_derive_v {
            ::std::option::Option::Some(true) => {
                __df_derive_values.set(__df_derive_leaf_idx, true);
            }
            ::std::option::Option::Some(false) => {}
            ::std::option::Option::None => {
                __df_derive_validity.set(__df_derive_leaf_idx, false);
            }
        }
        __df_derive_leaf_idx += 1;
    };
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_arr_inner = bool_leaf_array_tokens(pa_root, true, &values_ident, &validity_ident);
    let leaf_arr = idents::leaf_arr();
    let leaf_arr_expr = quote! {
        let #leaf_arr: #pa_root::array::BooleanArray = #leaf_arr_inner;
    };
    (storage, push, leaf_arr_expr)
}

/// Per-field local for the assembled Series — one per (field, depth)
/// combination, namespaced by `idx` so two adjacent fields don't collide.
fn vec_encoder_series_local(idx: usize) -> syn::Ident {
    idents::vec_field_series(idx)
}

/// Build the encoder for a `[Vec, ...]` shape: a self-contained columnar
/// block that allocates the per-field series local, renames it to the
/// field's column name, and pushes it onto the call site's `columns` vec.
/// The block is scoped so the per-field intermediate buffers (offsets vecs,
/// validity bitmaps, the field Series itself) are confined to the field's
/// scope, matching the pre-Step-4 emission shape.
fn vec_encoder(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecShape,
    leaf_dtype: &TokenStream,
) -> Encoder {
    let series_local = vec_encoder_series_local(ctx.idx);
    let decl = vec_emit_decl(ctx, spec, shape, leaf_dtype);
    let name = ctx.name;
    let columnar = quote! {
        {
            #decl
            let __df_derive_named = #series_local.with_name(#name.into());
            columns.push(__df_derive_named.into());
        }
    };
    Encoder::Multi { columnar }
}

/// Bare-bool variant of the vec encoder. At depth 1 with no inner-Option and
/// no outer-Option layers, uses `BooleanArray::from_slice` (bulk, no
/// bit-packing). For deeper or option-bearing shapes, routes through the
/// generalized `vec_encoder` with `VecLeafSpec::BoolBare` (a bit-packed
/// `MutableBitmap` set per element).
fn vec_encoder_bool_bare(ctx: &LeafCtx<'_>, shape: &VecShape) -> Encoder {
    if shape.depth() == 1 && !shape.any_outer_validity() {
        let pa_root = crate::codegen::polars_paths::polars_arrow_root();
        let pp = crate::codegen::polars_paths::prelude();
        let series_local = vec_encoder_series_local(ctx.idx);
        let body = bool_bare_depth1_body(ctx.access, &pa_root, &pp);
        let name = ctx.name;
        let decl = quote! { let #series_local: #pp::Series = { #body }; };
        let columnar = quote! {
            {
                #decl
                let __df_derive_named = #series_local.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }
        };
        return Encoder::Multi { columnar };
    }
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    vec_encoder(ctx, &VecLeafSpec::BoolBare, shape, &leaf_dtype)
}

/// `Vec<bool>` body: `Vec::extend` per outer row into a flat `Vec<bool>`,
/// then `BooleanArray::from_slice` at the end. No bit-packing because
/// `from_slice` is bulk and faster than `set` for the all-non-null case.
fn bool_bare_depth1_body(
    access: &TokenStream,
    pa_root: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let inner_offsets = idents::bool_inner_offsets();
    let total_leaves = idents::total_leaves();
    let it = idents::populator_iter();
    let leaf_arr = idents::leaf_arr();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    quote! {
        let mut #total_leaves: usize = 0;
        for #it in items {
            #total_leaves += (&(#access)).len();
        }
        let mut __df_derive_flat: ::std::vec::Vec<bool> =
            ::std::vec::Vec::with_capacity(#total_leaves);
        let mut #inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #inner_offsets.push(0);
        for #it in items {
            __df_derive_flat.extend((&(#access)).iter().copied());
            #inner_offsets.push(__df_derive_flat.len() as i64);
        }
        let #leaf_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::from_slice(&__df_derive_flat);
        let __df_derive_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#inner_offsets)?;
        let __df_derive_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&#leaf_arr).clone(),
            ),
            __df_derive_offsets_buf,
            ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_list_arr,
            #leaf_dtype,
        )
    }
}

/// `Vec<...>` (`as_str` transform) — same MBVA-based encoder as the bare
/// `String` path, but the value expression sources `&str` via UFCS through
/// `AsRef<str>`. The bytes are copied into the view array once, identical
/// to the `String::as_str()` path.
fn vec_encoder_as_str(ctx: &LeafCtx<'_>, shape: &VecShape, base: &StringyBase<'_>) -> Encoder {
    // For bare `String`, `&String` deref-coerces to `&str`; for non-String
    // bases we go through UFCS so generic-parameter and concrete-struct
    // leaves both resolve. `StringyBase` already encodes the parser's
    // accept set (String/Struct/Generic), so the match below is exhaustive
    // by type rather than by wildcard.
    let value_expr = match base {
        StringyBase::String => quote! { __df_derive_v.as_str() },
        StringyBase::Struct { ident, args } => {
            let ty_path = super::build_type_path(ident, *args);
            quote! { <#ty_path as ::core::convert::AsRef<str>>::as_ref(__df_derive_v) }
        }
        StringyBase::Generic(ident) => {
            quote! { <#ident as ::core::convert::AsRef<str>>::as_ref(__df_derive_v) }
        }
    };
    let spec = VecLeafSpec::StringLike {
        value_expr,
        extra_decls: Vec::new(),
    };
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

// --- Top-level dispatcher pieces ---

/// Build the depth-N `vec(inner)` encoder for every leaf shape. `LeafShape`
/// already encodes the parser's accept set, so the match below is
/// exhaustive by construction — no `_ => unreachable!()` arms.
///
/// Covers: bare numeric, `ISize`/`USize` (widened to `i64`/`u64` at the leaf
/// push site), `String`, `Bool`, `Decimal` (with `DecimalToInt128`),
/// `DateTime` (with `DateTimeToInt`), `as_str` borrow, and `to_string`.
pub(super) fn try_build_vec_encoder(
    shape: &LeafShape<'_>,
    ctx: &LeafCtx<'_>,
    vec_shape: &VecShape,
) -> Encoder {
    match shape {
        LeafShape::Numeric(base) => {
            let info = crate::codegen::type_registry::numeric_info(base)
                .expect("LeafShape::Numeric carries a numeric BaseType");
            // The loop binding is `&T` for Copy primitives, so dereferencing
            // produces the storage value directly. Bare and inner-Option
            // arms share the same expression because `build_vec_leaf_pieces`
            // re-binds the bare loop variable as `__df_derive_v` before
            // splicing the value expression in.
            let spec = VecLeafSpec::Numeric {
                native: info.native.clone(),
                value_expr: quote! { *__df_derive_v },
                needs_decimal_import: false,
            };
            vec_encoder(ctx, &spec, vec_shape, &info.dtype)
        }
        LeafShape::NumericWidened(base) => {
            // `ISize`/`USize` widen to `i64`/`u64` at the leaf push site.
            // The loop binding is `&isize`/`&usize`, so the cast reads the
            // pointed-to value first (`*v`) then widens to the target.
            let info = crate::codegen::type_registry::numeric_info(base)
                .expect("LeafShape::NumericWidened carries an `ISize`/`USize` BaseType");
            let target = info.native.clone();
            let spec = VecLeafSpec::Numeric {
                native: info.native.clone(),
                value_expr: quote! { (*__df_derive_v as #target) },
                needs_decimal_import: false,
            };
            vec_encoder(ctx, &spec, vec_shape, &info.dtype)
        }
        LeafShape::String => {
            let pp = crate::codegen::polars_paths::prelude();
            let leaf_dtype = quote! { #pp::DataType::String };
            let spec = VecLeafSpec::StringLike {
                value_expr: quote! { __df_derive_v.as_str() },
                extra_decls: Vec::new(),
            };
            vec_encoder(ctx, &spec, vec_shape, &leaf_dtype)
        }
        LeafShape::Bool => {
            if vec_shape.has_inner_option() {
                let pp = crate::codegen::polars_paths::prelude();
                let leaf_dtype = quote! { #pp::DataType::Boolean };
                vec_encoder(ctx, &VecLeafSpec::Bool, vec_shape, &leaf_dtype)
            } else {
                vec_encoder_bool_bare(ctx, vec_shape)
            }
        }
        LeafShape::DateTime(unit) => vec_encoder_datetime(ctx, *unit, vec_shape),
        LeafShape::Decimal { precision, scale } => {
            vec_encoder_decimal(ctx, *precision, *scale, vec_shape)
        }
        LeafShape::AsString => vec_encoder_to_string(ctx, vec_shape),
        // `as_str` borrow path: same MBVA-based encoder as `String`, but
        // the value expression goes through UFCS (`AsRef<str>`) instead of
        // `String::as_str`. Bytes are copied into the view array once.
        LeafShape::AsStr(stringy) => vec_encoder_as_str(ctx, vec_shape, stringy),
    }
}

fn vec_encoder_datetime(ctx: &LeafCtx<'_>, unit: DateTimeUnit, shape: &VecShape) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let unit_tokens = match unit {
        DateTimeUnit::Milliseconds => quote! { #pp::TimeUnit::Milliseconds },
        DateTimeUnit::Microseconds => quote! { #pp::TimeUnit::Microseconds },
        DateTimeUnit::Nanoseconds => quote! { #pp::TimeUnit::Nanoseconds },
    };
    let leaf_dtype = quote! {
        #pp::DataType::Datetime(#unit_tokens, ::std::option::Option::None)
    };
    let mapped_v = crate::codegen::type_registry::map_primitive_expr(
        &quote! { __df_derive_v },
        Some(&PrimitiveTransform::DateTimeToInt(unit)),
        ctx.decimal128_encode_trait,
    );
    let spec = VecLeafSpec::Numeric {
        native: quote! { i64 },
        value_expr: mapped_v,
        needs_decimal_import: false,
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

fn vec_encoder_decimal(ctx: &LeafCtx<'_>, precision: u8, scale: u8, shape: &VecShape) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let p = precision as usize;
    let s = scale as usize;
    let leaf_dtype = quote! { #pp::DataType::Decimal(#p, #s) };
    let mapped_v = crate::codegen::type_registry::map_primitive_expr(
        &quote! { __df_derive_v },
        Some(&PrimitiveTransform::DecimalToInt128 { precision, scale }),
        ctx.decimal128_encode_trait,
    );
    let spec = VecLeafSpec::Numeric {
        native: quote! { i128 },
        value_expr: mapped_v,
        needs_decimal_import: true,
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

fn vec_encoder_to_string(ctx: &LeafCtx<'_>, shape: &VecShape) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    // `to_string` materializes via `Display::fmt` into a reusable `String`
    // scratch — we splice that scratch's `as_str()` into the MBVA-push
    // expression, so the per-element work allocates the scratch once at
    // decl time and reuses on every row.
    let scratch = idents::primitive_str_scratch(ctx.idx);
    let value_expr = quote! {{
        use ::std::fmt::Write as _;
        #scratch.clear();
        ::std::write!(&mut #scratch, "{}", __df_derive_v).unwrap();
        #scratch.as_str()
    }};
    let spec = VecLeafSpec::StringLike {
        value_expr,
        extra_decls: vec![
            quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); },
        ],
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

/// Build the leaf-encoder bundle for a primitive shape. `LeafShape` encodes
/// the parser's accept set, so this dispatch is exhaustive by construction —
/// every "cannot reach this combination" check lives at
/// `LeafShape::from_base_transform` instead. Both `Numeric` and
/// `NumericWidened` route to the same `numeric_leaf` builder; the merged
/// `numeric_info` carries the widening info inline, so the two parser-time
/// `LeafShape` provenances yield distinct push tokens without needing
/// distinct dispatcher arms.
pub(super) fn build_leaf(shape: &LeafShape<'_>, ctx: &LeafCtx<'_>) -> LeafSpec {
    match shape {
        LeafShape::Numeric(base) | LeafShape::NumericWidened(base) => leaf::numeric_leaf(ctx, base),
        LeafShape::String => leaf::string_leaf(ctx),
        LeafShape::Bool => leaf::bool_leaf(ctx),
        LeafShape::DateTime(unit) => leaf::datetime_leaf(ctx, *unit),
        LeafShape::Decimal { precision, scale } => leaf::decimal_leaf(ctx, *precision, *scale),
        LeafShape::AsString => leaf::as_string_leaf(ctx),
        LeafShape::AsStr(stringy) => leaf::as_str_leaf(ctx, stringy),
    }
}
