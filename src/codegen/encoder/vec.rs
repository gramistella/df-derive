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

use crate::ir::{DateTimeUnit, LeafSpec, StringyBase, VecLayers};
use proc_macro2::TokenStream;
use quote::quote;

use super::emit::vec_emit_general;
use super::idents;
use super::leaf::{LeafBuilder, validity_into_option};
use super::leaf_kind::{LeafKind, PerElementPush};
use super::shape_walk::LayerIdents;
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
///
/// Exposed `pub(super)` so the unified emitter in
/// [`super::emit::vec_emit_general`] can reuse the per-element-push path's
/// per-layer ident factory.
pub(super) fn vec_layer_idents(depth: usize) -> Vec<LayerIdents> {
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
        VecLeafSpec::StringLike { .. } => quote! { #view_buf.len() },
        VecLeafSpec::Bool | VecLeafSpec::BoolBare => quote! { #leaf_idx },
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
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let b = idents::bitmap_builder();
    let v = idents::leaf_value();
    let storage = quote! {
        let mut #values_ident: #pa_root::bitmap::MutableBitmap = {
            let mut #b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            #b.extend_constant(#leaf_capacity_expr, false);
            #b
        };
        let mut #leaf_idx: usize = 0;
    };
    let push = quote! {
        if *#v {
            #values_ident.set(#leaf_idx, true);
        }
        #leaf_idx += 1;
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
    let b = idents::bitmap_builder();
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    // Numeric / Decimal / DateTime: `Vec<#native>` flat values.
    // Inner-Option carries a parallel `MutableBitmap` pre-filled `true`.
    let storage = if has_inner_option {
        quote! {
            let mut #flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
            let mut #validity: #pa_root::bitmap::MutableBitmap = {
                let mut #b =
                    #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
                #b.extend_constant(#leaf_capacity_expr, true);
                #b
            };
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
        quote! {
            let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::new(
                    <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                    #flat.into(),
                    ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                        #validity,
                    ),
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
    let b = idents::bitmap_builder();
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
        storage_parts.push(quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap = {
                let mut #b =
                    #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
                #b.extend_constant(#leaf_capacity_expr, true);
                #b
            };
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
        quote! {
            let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf
                .freeze()
                .with_validity(
                    ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                        #validity,
                    ),
                );
        }
    } else {
        quote! {
            let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf.freeze();
        }
    };
    (storage, push, leaf_arr_expr)
}

fn bool_inner_option_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let b = idents::bitmap_builder();
    let v = idents::leaf_value();
    // Bool with inner-Option only — bare bool is served by
    // `vec_encoder_bool_bare`. Bit-packed values bitmap (pre-filled `false`)
    // + parallel validity bitmap (pre-filled `true`) + leaf index counter so
    // `set(i, ...)` lands on the right bit.
    let storage = quote! {
        let mut #values_ident: #pa_root::bitmap::MutableBitmap = {
            let mut #b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            #b.extend_constant(#leaf_capacity_expr, false);
            #b
        };
        let mut #validity_ident: #pa_root::bitmap::MutableBitmap = {
            let mut #b =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
            #b.extend_constant(#leaf_capacity_expr, true);
            #b
        };
        let mut #leaf_idx: usize = 0;
    };
    let push = quote! {
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
    let decl = vec_emit_general(&kind, ctx.base.access, ctx.base.idx, shape);
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
/// generalized `vec_encoder` with `VecLeafSpec::BoolBare` (a bit-packed
/// `MutableBitmap` set per element).
fn vec_encoder_bool_bare(ctx: &LeafCtx<'_>, shape: &VecLayers) -> Encoder {
    if shape.depth() == 1 && !shape.any_outer_validity() {
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
/// to the `String::as_str()` path.
fn vec_encoder_as_str(ctx: &LeafCtx<'_>, shape: &VecLayers, base: &StringyBase) -> Encoder {
    // For bare `String`, `&String` deref-coerces to `&str`; for non-String
    // bases we go through UFCS so generic-parameter and concrete-struct
    // leaves both resolve. `StringyBase` already encodes the parser's
    // accept set (String/Struct/Generic), so the match below is exhaustive
    // by type rather than by wildcard.
    let v = idents::leaf_value();
    let value_expr = match base {
        StringyBase::String => quote! { #v.as_str() },
        StringyBase::Struct(ident, args) => {
            let ty_path = super::build_type_path(ident, args.as_ref());
            quote! { <#ty_path as ::core::convert::AsRef<str>>::as_ref(#v) }
        }
        StringyBase::Generic(ident) => {
            quote! { <#ident as ::core::convert::AsRef<str>>::as_ref(#v) }
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
            let value_expr = if kind.is_widened() {
                let target = &info.native;
                quote! { (*#v as #target) }
            } else {
                quote! { *#v }
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
        LeafSpec::Decimal { precision, scale } => {
            vec_encoder_decimal(ctx, *precision, *scale, vec_shape)
        }
        LeafSpec::AsString => vec_encoder_to_string(ctx, vec_shape),
        // `as_str` borrow path: same MBVA-based encoder as `String`, but
        // the value expression goes through UFCS (`AsRef<str>`) instead of
        // `String::as_str`. Bytes are copied into the view array once.
        LeafSpec::AsStr(stringy) => vec_encoder_as_str(ctx, vec_shape, stringy),
        LeafSpec::Struct(..) | LeafSpec::Generic(_) => unreachable!(
            "df-derive: try_build_vec_encoder reached with Struct/Generic leaf — \
             those route through build_nested_encoder via the FieldRoute split \
             in `super::strategy::classify_field`",
        ),
    }
}

fn vec_encoder_datetime(ctx: &LeafCtx<'_>, unit: DateTimeUnit, shape: &VecLayers) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let v = idents::leaf_value();
    let unit_tokens = match unit {
        DateTimeUnit::Milliseconds => quote! { #pp::TimeUnit::Milliseconds },
        DateTimeUnit::Microseconds => quote! { #pp::TimeUnit::Microseconds },
        DateTimeUnit::Nanoseconds => quote! { #pp::TimeUnit::Nanoseconds },
    };
    let leaf_dtype = quote! {
        #pp::DataType::Datetime(#unit_tokens, ::std::option::Option::None)
    };
    let leaf = LeafSpec::DateTime(unit);
    let mapped_v = crate::codegen::type_registry::map_primitive_expr(
        &quote! { #v },
        &leaf,
        ctx.decimal128_encode_trait,
    );
    let spec = VecLeafSpec::Numeric {
        native: quote! { i64 },
        value_expr: mapped_v,
        needs_decimal_import: false,
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
}

fn vec_encoder_decimal(ctx: &LeafCtx<'_>, precision: u8, scale: u8, shape: &VecLayers) -> Encoder {
    let pp = crate::codegen::polars_paths::prelude();
    let v = idents::leaf_value();
    let p = precision as usize;
    let s = scale as usize;
    let leaf_dtype = quote! { #pp::DataType::Decimal(#p, #s) };
    let leaf = LeafSpec::Decimal { precision, scale };
    let mapped_v = crate::codegen::type_registry::map_primitive_expr(
        &quote! { #v },
        &leaf,
        ctx.decimal128_encode_trait,
    );
    let spec = VecLeafSpec::Numeric {
        native: quote! { i128 },
        value_expr: mapped_v,
        needs_decimal_import: true,
    };
    vec_encoder(ctx, &spec, shape, &leaf_dtype)
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

/// Build the leaf-encoder bundle for a primitive `LeafSpec`. Total over the
/// primitive variants of `LeafSpec` — `Struct`/`Generic` cannot reach this
/// dispatcher because `super::strategy::classify_field` routes them through
/// `build_nested_encoder`. The fixed-width and `ISize`/`USize` numeric
/// variants both route to `numeric_leaf`; the widening info is carried
/// inline on `NumericKind`, so the two provenances yield distinct push
/// tokens without needing distinct dispatcher arms.
pub(super) fn build_leaf(leaf: &LeafSpec, ctx: &LeafCtx<'_>) -> LeafBuilder {
    match leaf {
        LeafSpec::Numeric(kind) => leaf::numeric_leaf(ctx, *kind),
        LeafSpec::String => leaf::string_leaf(ctx),
        LeafSpec::Bool => leaf::bool_leaf(ctx),
        LeafSpec::DateTime(unit) => leaf::datetime_leaf(ctx, *unit),
        LeafSpec::Decimal { precision, scale } => leaf::decimal_leaf(ctx, *precision, *scale),
        LeafSpec::AsString => leaf::as_string_leaf(ctx),
        LeafSpec::AsStr(stringy) => leaf::as_str_leaf(ctx, stringy),
        LeafSpec::Struct(..) | LeafSpec::Generic(_) => unreachable!(
            "df-derive: build_leaf reached with Struct/Generic leaf — those \
             route through build_nested_encoder via the FieldRoute split in \
             `super::strategy::classify_field`",
        ),
    }
}
