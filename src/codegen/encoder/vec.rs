//! Vec combinator + primitive vec dispatch + `build_leaf`.
//!
//! Implements the `vec(inner)` combinator that fuses N consecutive `Vec`
//! layers into a single bulk emission (one flat values buffer at the
//! deepest layer, one optional inner validity bitmap, one pair of offsets
//! per layer, one optional outer-list validity bitmap per layer that has
//! an adjoining `Option`). The bulk-fusion invariant lives here.
//!
//! Also hosts `build_leaf` — the dispatcher from a [`crate::ir::LeafSpec`]
//! to a primitive leaf encoder — because it sits at the leaf/vec boundary
//! and `try_build_vec_encoder` shares the leaf coverage matrix.

use crate::ir::{DateTimeUnit, DurationSource, LeafSpec, StringyBase, VecLayers, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;

use super::emit::vec_emit_general;
use super::idents;
use super::leaf::{LeafArm, LeafArmKind, mb_decl_filled, validity_into_option};
use super::leaf_kind::{LeafKind, PerElementPush};
use super::{Encoder, LeafCtx, leaf};

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
/// regardless of context — the only knob is `has_inner_option`, which is
/// already passed alongside the spec. The depth-1 bare-bool fast path
/// (flat `Vec<bool>` + `BooleanArray::from_slice`) is served by a
/// dedicated builder ahead of this enum, not by an extra variant.
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
    /// `Binary` (`#[df_derive(as_binary)]` over `Vec<Vec<u8>>` /
    /// `Vec<Option<Vec<u8>>>`) over `MutableBinaryViewArray<[u8]>`.
    /// `value_expr` materializes an `AsRef<[u8]>` from the `__df_derive_v`
    /// binding (the loop binds `&Vec<u8>` so the expression is just
    /// `&#v[..]`). Parallel to `StringLike` but freezes a `BinaryViewArray`
    /// instead of a `Utf8ViewArray`.
    BinaryLike { value_expr: TokenStream },
    /// Bool over a bit-packed `MutableBitmap` values buffer. The
    /// `has_inner_option` flag selects the inner-Option layout (parallel
    /// validity bitmap pre-filled `true`) versus the bare layout
    /// (no validity bitmap). Depth-1 bare-bool with no outer-Option layer
    /// uses a flat `Vec<bool>` fast path and is served by
    /// `vec_encoder_bool_bare` directly, not this enum.
    Bool,
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

/// The expression that becomes `<offsets>.push(<expr> as i64)` at the
/// inner-vec-loop tail. Numeric / decimal / datetime use the flat-vec
/// length; the string-view path uses the MBVA `len()` (which is the count
/// of pushed views); the bool bit-packed-bitmap layout tracks a per-leaf
/// index so it can stay in sync with both the values and validity bitmaps.
fn leaf_offsets_post_push_tokens(spec: &VecLeafSpec) -> TokenStream {
    let flat = idents::vec_flat();
    let view_buf = idents::vec_view_buf();
    let leaf_idx = idents::vec_leaf_idx();
    match spec {
        VecLeafSpec::Numeric { .. } => quote! { #flat.len() },
        VecLeafSpec::StringLike { .. } | VecLeafSpec::BinaryLike { .. } => {
            quote! { #view_buf.len() }
        }
        VecLeafSpec::Bool => quote! { #leaf_idx },
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
    inner_derefs: usize,
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
        VecLeafSpec::BinaryLike { value_expr } => {
            binary_like_leaf_pieces(value_expr, has_inner_option, leaf_capacity_expr, pa_root)
        }
        VecLeafSpec::Bool => {
            if has_inner_option {
                bool_inner_option_leaf_pieces(leaf_capacity_expr, pa_root, inner_derefs)
            } else {
                bool_bare_leaf_pieces(leaf_capacity_expr, pa_root, inner_derefs)
            }
        }
    }
}

/// Bit-packed values bitmap pre-filled `false`, with a per-leaf index
/// counter for `set(idx, true)`. Used by deeper-than-1 / outer-Option-bearing
/// `Vec<bool>` shapes that don't qualify for the depth-1 `from_slice`
/// fast path. The leaf-array build skips the validity argument.
///
/// `inner_derefs` is the count of smart-pointer layers between the
/// outermost Vec wrapper and the bool leaf (e.g. `Vec<Box<bool>>` is 1).
/// The push body applies `1 + inner_derefs` derefs to the loop binding —
/// the first to unwrap the `&` from the for-loop, the rest to walk the
/// smart-pointer chain.
fn bool_bare_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
    inner_derefs: usize,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let v = idents::leaf_value();
    let values_decl = mb_decl_filled(&values_ident, leaf_capacity_expr, false);
    let storage = quote! {
        #values_decl
        let mut #leaf_idx: usize = 0;
    };
    // No smart pointers: keep the original `*v` token shape. With smart
    // pointers, walk the chain via parenthesized derefs.
    let push = if inner_derefs == 0 {
        quote! {
            if *#v {
                #values_ident.set(#leaf_idx, true);
            }
            #leaf_idx += 1;
        }
    } else {
        let v_chain = leaf::apply_inner_derefs(&quote! { #v }, 1 + inner_derefs);
        quote! {
            if #v_chain {
                #values_ident.set(#leaf_idx, true);
            }
            #leaf_idx += 1;
        }
    };
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
    let flat = idents::vec_flat();
    let validity = idents::bool_validity();
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    // Numeric / Decimal / DateTime: `Vec<#native>` flat values.
    // Inner-Option carries a parallel `MutableBitmap` pre-filled `true`.
    let storage = if has_inner_option {
        let validity_decl = mb_decl_filled(&validity, leaf_capacity_expr, true);
        quote! {
            let mut #flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
            #validity_decl
        }
    } else {
        quote! {
            let mut #flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
        }
    };
    // The bare and inner-Option arms both reference `__df_derive_v` — the
    // bare arm gets it from the loop binding directly (the for-loop in
    // `emit::pep_leaf_body` binds the leaf as `__df_derive_v`), the option
    // arm gets it from the `Some(v)` pattern. Sharing one `value_expr`
    // avoids two near-duplicate per-spec expressions.
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
            match #v {
                ::std::option::Option::Some(#v) => {
                    #flat.push({ #value_expr });
                }
                ::std::option::Option::None => {
                    #flat.push(<#native as ::std::default::Default>::default());
                    #validity.set(#flat.len() - 1, false);
                }
            }
        }
    } else {
        quote! {
            #flat.push(#value_expr);
        }
    };
    let leaf_arr_expr = if has_inner_option {
        let valid_opt = validity_into_option(&validity);
        quote! {
            let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::new(
                    <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                    #flat.into(),
                    #valid_opt,
                );
        }
    } else {
        quote! {
            let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::from_vec(#flat);
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
    let view_buf = idents::vec_view_buf();
    let validity = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    // String / to_string / as_str: `MutableBinaryViewArray<str>` flat values.
    // Inner-Option uses a separate validity bitmap pre-filled `true`, plus
    // a row index that advances per element so `validity.set(i, false)`
    // hits the right bit on `None`.
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    for d in extra_decls {
        storage_parts.push(d.clone());
    }
    storage_parts.push(quote! {
        let mut #view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        let validity_decl = mb_decl_filled(&validity, leaf_capacity_expr, true);
        storage_parts.push(quote! {
            #validity_decl
            let mut #leaf_idx: usize = 0;
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let push = if has_inner_option {
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #view_buf.push_value_ignore_validity({ #value_expr });
                }
                ::std::option::Option::None => {
                    #view_buf.push_value_ignore_validity("");
                    #validity.set(#leaf_idx, false);
                }
            }
            #leaf_idx += 1;
        }
    } else {
        quote! {
            #view_buf.push_value_ignore_validity({ #value_expr });
        }
    };
    let leaf_arr_expr = if has_inner_option {
        let valid_opt = validity_into_option(&validity);
        quote! {
            let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf
                .freeze()
                .with_validity(#valid_opt);
        }
    } else {
        quote! {
            let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf.freeze();
        }
    };
    (storage, push, leaf_arr_expr)
}

/// Byte-blob analogue of [`string_like_leaf_pieces`]. Accumulates into
/// `MutableBinaryViewArray<[u8]>` and freezes into `BinaryViewArray`. The
/// `None` arm pushes an empty slice (`&[][..]`) so the bitmap-pair layout
/// matches `string_like_leaf_pieces` byte-for-byte modulo the leaf type.
fn binary_like_leaf_pieces(
    value_expr: &TokenStream,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let view_buf = idents::vec_view_buf();
    let validity = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    storage_parts.push(quote! {
        let mut #view_buf: #pa_root::array::MutableBinaryViewArray<[u8]> =
            #pa_root::array::MutableBinaryViewArray::<[u8]>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        let validity_decl = mb_decl_filled(&validity, leaf_capacity_expr, true);
        storage_parts.push(quote! {
            #validity_decl
            let mut #leaf_idx: usize = 0;
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let empty = quote! { &[][..] };
    let push = if has_inner_option {
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #view_buf.push_value_ignore_validity({ #value_expr });
                }
                ::std::option::Option::None => {
                    #view_buf.push_value_ignore_validity(#empty);
                    #validity.set(#leaf_idx, false);
                }
            }
            #leaf_idx += 1;
        }
    } else {
        quote! {
            #view_buf.push_value_ignore_validity({ #value_expr });
        }
    };
    let leaf_arr_expr = if has_inner_option {
        let valid_opt = validity_into_option(&validity);
        quote! {
            let #leaf_arr: #pa_root::array::BinaryViewArray = #view_buf
                .freeze()
                .with_validity(#valid_opt);
        }
    } else {
        quote! {
            let #leaf_arr: #pa_root::array::BinaryViewArray = #view_buf.freeze();
        }
    };
    (storage, push, leaf_arr_expr)
}

fn bool_inner_option_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
    inner_derefs: usize,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let v = idents::leaf_value();
    // Bool with inner-Option only — bare bool is served by
    // `vec_encoder_bool_bare`. Bit-packed values bitmap (pre-filled `false`)
    // + parallel validity bitmap (pre-filled `true`) + leaf index counter so
    // `set(i, ...)` lands on the right bit.
    let values_decl = mb_decl_filled(&values_ident, leaf_capacity_expr, false);
    let validity_decl = mb_decl_filled(&validity_ident, leaf_capacity_expr, true);
    let storage = quote! {
        #values_decl
        #validity_decl
        let mut #leaf_idx: usize = 0;
    };
    // The Option-bind here is `Some(v)` against the value bound by the
    // outer for loop (`v: &Option<...>`-or-`Option<&...>` depending on the
    // wrapper layout). For `Vec<Option<Box<bool>>>` the bound `v` is
    // `Box<bool>`, requiring `inner_derefs` extra `*` to reach the bool.
    // The match itself stays on `Some(true)/Some(false)` literals when
    // there are no inner smart pointers; otherwise the inner is a non-bool
    // type (the smart pointer) and we bind a name + nested match instead.
    let push = if inner_derefs == 0 {
        quote! {
            match #v {
                ::std::option::Option::Some(true) => {
                    #values_ident.set(#leaf_idx, true);
                }
                ::std::option::Option::Some(false) => {}
                ::std::option::Option::None => {
                    #validity_ident.set(#leaf_idx, false);
                }
            }
            #leaf_idx += 1;
        }
    } else {
        // Bind the Some payload as `__df_derive_v` (shadowing the outer
        // loop binding inside this arm only) so apply_inner_derefs sees a
        // simple ident expression. The outer `v` is the Option-typed
        // value; the inner `v` is the smart-pointer-wrapped bool we need
        // to deref.
        let inner_v = v.clone();
        let v_deref = leaf::apply_inner_derefs(&quote! { #inner_v }, inner_derefs);
        quote! {
            match #v {
                ::std::option::Option::Some(#inner_v) => {
                    if #v_deref {
                        #values_ident.set(#leaf_idx, true);
                    }
                }
                ::std::option::Option::None => {
                    #validity_ident.set(#leaf_idx, false);
                }
            }
            #leaf_idx += 1;
        }
    };
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
///
/// Lowers `VecLeafSpec` (the per-element-push leaf description local to
/// this file) into [`LeafKind::PerElementPush`] and routes through the
/// unified [`vec_emit_general`]. The lowering is mechanical: the storage
/// decls / per-elem push / leaf-array build / offsets-push expression are
/// already shaped by `build_vec_leaf_pieces`; the leaf logical dtype and
/// optional decimal-trait import live in this file's encoder gateways
/// (`vec_encoder_decimal` etc) and ride into the unified emitter via the
/// `PerElementPush` payload.
fn vec_encoder(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecLayers,
    leaf_dtype: &TokenStream,
) -> Encoder {
    let series_local = vec_encoder_series_local(ctx.base.idx);
    let pep = lower_to_pep(ctx, spec, shape, leaf_dtype);
    let kind = LeafKind::PerElementPush(pep);
    let wrapper = WrapperShape::Vec(shape.clone());
    let decl = vec_emit_general(&kind, ctx.base.access, ctx.base.idx, &wrapper);
    let name = ctx.base.name;
    let named = idents::field_named_series();
    let columnar = quote! {
        {
            #decl
            let #named = #series_local.with_name(#name.into());
            columns.push(#named.into());
        }
    };
    Encoder::Multi { columnar }
}

/// Lower a `VecLeafSpec` into a [`PerElementPush`] payload the unified
/// emitter consumes. The leaf-capacity expression is `__df_derive_total_leaves`
/// (the precount loop's leaf-element accumulator); `build_vec_leaf_pieces`
/// returns storage decls keyed off it. The decimal-trait import is the
/// only `extra_imports` payload used today (the `Decimal` leaf needs the
/// `Decimal128Encode` trait in scope so `try_to_i128_mantissa` resolves
/// via dot syntax).
fn lower_to_pep(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecLayers,
    leaf_dtype: &TokenStream,
) -> PerElementPush {
    let pa_root = crate::codegen::polars_paths::polars_arrow_root();
    let total_leaves = idents::total_leaves();
    let leaf_capacity_expr = quote! { #total_leaves };
    let (leaf_storage_decls, per_elem_push, leaf_arr_expr) = build_vec_leaf_pieces(
        spec,
        shape.has_inner_option(),
        &leaf_capacity_expr,
        &pa_root,
        ctx.inner_smart_ptr_depth,
    );
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
    PerElementPush {
        per_elem_push,
        storage_decls: leaf_storage_decls,
        leaf_arr_expr,
        leaf_offsets_post_push,
        extra_imports,
        leaf_logical_dtype: leaf_dtype.clone(),
    }
}

/// Bare-bool variant of the vec encoder. At depth 1 with no inner-Option and
/// no outer-Option layers, uses `BooleanArray::from_slice` (bulk, no
/// bit-packing). For deeper or option-bearing shapes, routes through the
/// generalized `vec_encoder` with `VecLeafSpec::Bool` — the
/// `has_inner_option` flag (`false` here) selects the bare bit-packed
/// `MutableBitmap` layout inside `build_vec_leaf_pieces`.
fn vec_encoder_bool_bare(ctx: &LeafCtx<'_>, shape: &VecLayers) -> Encoder {
    // The depth-1 fast path uses `(&access).iter().copied()`, which
    // requires a `&bool`-yielding iterator. With inner smart pointers
    // (`Vec<Box<bool>>` and friends) the iter yields `&Box<bool>` etc., so
    // `.copied()` fails. Route those shapes through the bit-packed
    // bitmap path which threads `inner_derefs` into the per-element push.
    if shape.depth() == 1 && !shape.any_outer_validity() && ctx.inner_smart_ptr_depth == 0 {
        let pa_root = crate::codegen::polars_paths::polars_arrow_root();
        let pp = crate::codegen::polars_paths::prelude();
        let series_local = vec_encoder_series_local(ctx.base.idx);
        let body = bool_bare_depth1_body(ctx.base.access, &pa_root, &pp);
        let name = ctx.base.name;
        let named = idents::field_named_series();
        let decl = quote! { let #series_local: #pp::Series = { #body }; };
        let columnar = quote! {
            {
                #decl
                let #named = #series_local.with_name(#name.into());
                columns.push(#named.into());
            }
        };
        return Encoder::Multi { columnar };
    }
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    vec_encoder(ctx, &VecLeafSpec::Bool, shape, &leaf_dtype)
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
    let flat = idents::vec_flat();
    let offsets_buf = idents::bool_bare_offsets_buf();
    let list_arr = idents::bool_bare_list_arr();
    let assemble_helper = idents::assemble_helper();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    quote! {
        let mut #total_leaves: usize = 0;
        for #it in items {
            #total_leaves += (&(#access)).len();
        }
        let mut #flat: ::std::vec::Vec<bool> =
            ::std::vec::Vec::with_capacity(#total_leaves);
        let mut #inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #inner_offsets.push(0);
        for #it in items {
            #flat.extend((&(#access)).iter().copied());
            #inner_offsets.push(#flat.len() as i64);
        }
        let #leaf_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::from_slice(&#flat);
        let #offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#inner_offsets)?;
        let #list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&#leaf_arr).clone(),
            ),
            #offsets_buf,
            ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        #assemble_helper(
            #list_arr,
            #leaf_dtype,
        )
    }
}

/// `Vec<...>` (`as_str` transform) — same MBVA-based encoder as the bare
/// `String` path, but the value expression sources `&str` via UFCS through
/// `AsRef<str>`. The bytes are copied into the view array once, identical
/// to the `String::as_str()` path. The `String`-vs-UFCS branch is shared
/// with the single-Option leaf and the multi-Option wrapper through
/// [`super::stringy_value_expr`].
fn vec_encoder_as_str(ctx: &LeafCtx<'_>, shape: &VecLayers, base: &StringyBase) -> Encoder {
    let v = idents::leaf_value();
    let value_expr =
        super::stringy_value_expr(base, &quote! { #v }, super::StringyExprKind::MbvaValue);
    let spec = VecLeafSpec::StringLike {
        value_expr,
        extra_decls: Vec::new(),
    };
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

// --- Top-level dispatcher pieces ---

/// Shared `unreachable!` body for primitive-only leaf dispatchers
/// (`build_leaf`, `try_build_vec_encoder`). `Struct` / `Generic` leaves
/// never reach these dispatchers because `super::strategy::LeafSpec::route`
/// routes them through `build_nested_encoder` instead.
fn unreachable_struct_in_primitive_dispatcher(fn_name: &str) -> ! {
    unreachable!(
        "df-derive: {fn_name} reached with Struct/Generic leaf — those \
         route through build_nested_encoder via the FieldRoute split in \
         `super::strategy::LeafSpec::route`",
    )
}

/// Build the depth-N `vec(inner)` encoder for every leaf shape. The
/// [`LeafSpec`] already encodes the parser's accept set, so the match
/// below is exhaustive by construction — `Struct`/`Generic` leaves never
/// reach this dispatcher (they route through `build_nested_encoder`).
///
/// Covers: bare numeric, `ISize`/`USize` (widened to `i64`/`u64` at the leaf
/// push site), `String`, `Bool`, `Decimal`, `DateTime`, `as_str` borrow,
/// and `to_string`.
pub(super) fn try_build_vec_encoder(
    leaf: &LeafSpec,
    ctx: &LeafCtx<'_>,
    vec_shape: &VecLayers,
) -> Encoder {
    let v = idents::leaf_value();
    let inner_derefs = ctx.inner_smart_ptr_depth;
    match leaf {
        LeafSpec::Numeric(kind) => {
            let info = crate::codegen::type_registry::numeric_info_for(*kind);
            // The loop binding is `&T` for Copy primitives, so dereferencing
            // produces the storage value directly. Bare and inner-Option
            // arms share the same expression because `build_vec_leaf_pieces`
            // re-binds the bare loop variable as `__df_derive_v` before
            // splicing the value expression in.
            //
            // `ISize`/`USize` widen to `i64`/`u64` at the leaf push site.
            // The loop binding is `&isize`/`&usize`, so the cast reads the
            // pointed-to value first (`*v`) then widens to the target.
            //
            // Inner smart pointers (`Vec<Box<T>>`, `Vec<Arc<T>>`, etc.)
            // contribute extra derefs after the loop-binding `&` is
            // unwrapped. The bare path keeps the original `*v` shape (no
            // extra parens) — bench `08_complex_wrappers` is sensitive to
            // the exact token shape; switching to `(*v)` for the
            // zero-smart-pointer case shifted timings by ~5-10% in
            // prototypes. With smart pointers we use `apply_inner_derefs`
            // because the parens are needed to compose multiple derefs.
            let value_expr = if inner_derefs == 0 {
                if kind.is_widened() {
                    let target = &info.native;
                    quote! { (*#v as #target) }
                } else {
                    quote! { *#v }
                }
            } else {
                let v_chain = leaf::apply_inner_derefs(&quote! { #v }, 1 + inner_derefs);
                if kind.is_widened() {
                    let target = &info.native;
                    quote! { (#v_chain as #target) }
                } else {
                    quote! { #v_chain }
                }
            };
            let spec = VecLeafSpec::Numeric {
                native: info.native.clone(),
                value_expr,
                needs_decimal_import: false,
            };
            vec_encoder(ctx, &spec, vec_shape, &info.dtype)
        }
        LeafSpec::String => {
            let pp = crate::codegen::polars_paths::prelude();
            let leaf_dtype = quote! { #pp::DataType::String };
            let spec = VecLeafSpec::StringLike {
                value_expr: quote! { #v.as_str() },
                extra_decls: Vec::new(),
            };
            vec_encoder(ctx, &spec, vec_shape, &leaf_dtype)
        }
        LeafSpec::Binary => {
            // The loop binds `&Vec<u8>` (or `&Option<Vec<u8>>` in the
            // inner-Option arm, then `Some(v)` rebinds to `&Vec<u8>`); slicing
            // produces a `&[u8]` accepted by `MutableBinaryViewArray<[u8]>`.
            let pp = crate::codegen::polars_paths::prelude();
            let leaf_dtype = quote! { #pp::DataType::Binary };
            let spec = VecLeafSpec::BinaryLike {
                value_expr: quote! { &#v[..] },
            };
            vec_encoder(ctx, &spec, vec_shape, &leaf_dtype)
        }
        LeafSpec::Bool => {
            if vec_shape.has_inner_option() {
                let pp = crate::codegen::polars_paths::prelude();
                let leaf_dtype = quote! { #pp::DataType::Boolean };
                vec_encoder(ctx, &VecLeafSpec::Bool, vec_shape, &leaf_dtype)
            } else {
                vec_encoder_bool_bare(ctx, vec_shape)
            }
        }
        LeafSpec::DateTime(unit) => vec_encoder_datetime(ctx, *unit, vec_shape),
        LeafSpec::NaiveDate => vec_encoder_naive_date(ctx, vec_shape),
        LeafSpec::NaiveTime => vec_encoder_naive_time(ctx, vec_shape),
        LeafSpec::Duration { unit, source } => vec_encoder_duration(ctx, *unit, *source, vec_shape),
        LeafSpec::Decimal { precision, scale } => {
            vec_encoder_decimal(ctx, *precision, *scale, vec_shape)
        }
        LeafSpec::AsString => vec_encoder_to_string(ctx, vec_shape),
        // `as_str` borrow path: same MBVA-based encoder as `String`, but
        // the value expression goes through UFCS (`AsRef<str>`) instead of
        // `String::as_str`. Bytes are copied into the view array once.
        LeafSpec::AsStr(stringy) => vec_encoder_as_str(ctx, vec_shape, stringy),
        LeafSpec::Struct(..) | LeafSpec::Generic(_) => {
            unreachable_struct_in_primitive_dispatcher("try_build_vec_encoder")
        }
        LeafSpec::Tuple(_) => unreachable!(
            "df-derive: try_build_vec_encoder reached with Tuple leaf — tuple \
             fields route through the tuple emitter, not the primitive vec \
             dispatcher",
        ),
    }
}

/// Shared body of `vec_encoder_datetime` / `vec_encoder_decimal`: build a
/// `VecLeafSpec::Numeric` wrapping a per-row mapped value (chrono epoch i64
/// for `DateTime`, mantissa i128 for `Decimal`). The leaf-specific pieces
/// (`leaf` for `map_primitive_expr` dispatch, `native`, `leaf_dtype`,
/// `needs_decimal_import`) are passed in.
fn vec_encoder_mapped_numeric(
    ctx: &LeafCtx<'_>,
    shape: &VecLayers,
    leaf: &LeafSpec,
    native: TokenStream,
    leaf_dtype: TokenStream,
    needs_decimal_import: bool,
) -> Encoder {
    let v = idents::leaf_value();
    let mapped_v = crate::codegen::type_registry::map_primitive_expr(
        &quote! { #v },
        leaf,
        ctx.decimal128_encode_trait,
    );
    let spec = VecLeafSpec::Numeric {
        native,
        value_expr: mapped_v,
        needs_decimal_import,
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

fn vec_encoder_datetime(ctx: &LeafCtx<'_>, unit: DateTimeUnit, shape: &VecLayers) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let unit_tokens = crate::codegen::type_registry::time_unit_tokens(unit);
    let leaf_dtype = quote! {
        #pp::DataType::Datetime(#unit_tokens, ::std::option::Option::None)
    };
    vec_encoder_mapped_numeric(
        ctx,
        shape,
        &LeafSpec::DateTime(unit),
        quote! { i64 },
        leaf_dtype,
        false,
    )
}

fn vec_encoder_naive_date(ctx: &LeafCtx<'_>, shape: &VecLayers) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Date };
    vec_encoder_mapped_numeric(
        ctx,
        shape,
        &LeafSpec::NaiveDate,
        quote! { i32 },
        leaf_dtype,
        false,
    )
}

fn vec_encoder_naive_time(ctx: &LeafCtx<'_>, shape: &VecLayers) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Time };
    vec_encoder_mapped_numeric(
        ctx,
        shape,
        &LeafSpec::NaiveTime,
        quote! { i64 },
        leaf_dtype,
        false,
    )
}

fn vec_encoder_duration(
    ctx: &LeafCtx<'_>,
    unit: DateTimeUnit,
    source: DurationSource,
    shape: &VecLayers,
) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let unit_tokens = crate::codegen::type_registry::time_unit_tokens(unit);
    let leaf_dtype = quote! { #pp::DataType::Duration(#unit_tokens) };
    vec_encoder_mapped_numeric(
        ctx,
        shape,
        &LeafSpec::Duration { unit, source },
        quote! { i64 },
        leaf_dtype,
        false,
    )
}

fn vec_encoder_decimal(ctx: &LeafCtx<'_>, precision: u8, scale: u8, shape: &VecLayers) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let p = precision as usize;
    let s = scale as usize;
    let leaf_dtype = quote! { #pp::DataType::Decimal(#p, #s) };
    vec_encoder_mapped_numeric(
        ctx,
        shape,
        &LeafSpec::Decimal { precision, scale },
        quote! { i128 },
        leaf_dtype,
        true,
    )
}

fn vec_encoder_to_string(ctx: &LeafCtx<'_>, shape: &VecLayers) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    // `to_string` materializes via `Display::fmt` into a reusable `String`
    // scratch — we splice that scratch's `as_str()` into the MBVA-push
    // expression, so the per-element work allocates the scratch once at
    // decl time and reuses on every row.
    let scratch = idents::primitive_str_scratch(ctx.base.idx);
    let v = idents::leaf_value();
    let value_expr = quote! {{
        use ::std::fmt::Write as _;
        #scratch.clear();
        ::std::write!(&mut #scratch, "{}", #v).unwrap();
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

/// Build a single arm of the leaf encoder for a primitive `LeafSpec`. The
/// `kind` parameter selects between the unwrapped (`Bare`) and `[Option]`
/// (`Option`) shapes — only the requested arm is constructed. Total over
/// the primitive variants of `LeafSpec` — `Struct`/`Generic` cannot reach
/// this dispatcher because `super::strategy::LeafSpec::route` routes them
/// through `build_nested_encoder`. The fixed-width and `ISize`/`USize`
/// numeric variants both route to `numeric_leaf`; the widening info is
/// carried inline on `NumericKind`, so the two provenances yield distinct
/// push tokens without needing distinct dispatcher arms.
pub(super) fn build_leaf(leaf: &LeafSpec, ctx: &LeafCtx<'_>, kind: LeafArmKind) -> LeafArm {
    match leaf {
        LeafSpec::Numeric(num_kind) => leaf::numeric_leaf(ctx, *num_kind, kind),
        LeafSpec::String => leaf::string_leaf(ctx, kind),
        LeafSpec::Bool => leaf::bool_leaf(ctx, kind),
        LeafSpec::Binary => leaf::binary_leaf(ctx, kind),
        LeafSpec::DateTime(unit) => leaf::datetime_leaf(ctx, *unit, kind),
        LeafSpec::NaiveDate => leaf::naive_date_leaf(ctx, kind),
        LeafSpec::NaiveTime => leaf::naive_time_leaf(ctx, kind),
        LeafSpec::Duration { unit, source } => leaf::duration_leaf(ctx, *unit, *source, kind),
        LeafSpec::Decimal { precision, scale } => leaf::decimal_leaf(ctx, *precision, *scale, kind),
        LeafSpec::AsString => leaf::as_string_leaf(ctx, kind),
        LeafSpec::AsStr(stringy) => leaf::as_str_leaf(ctx, stringy, kind),
        LeafSpec::Struct(..) | LeafSpec::Generic(_) => {
            unreachable_struct_in_primitive_dispatcher("build_leaf")
        }
        LeafSpec::Tuple(_) => unreachable!(
            "df-derive: build_leaf reached with Tuple leaf — tuple fields route \
             through the tuple emitter, not the primitive leaf dispatcher",
        ),
    }
}
