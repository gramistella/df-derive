//! Vec combinator and primitive leaf dispatch.
//!
//! `vec(inner)` fuses N consecutive `Vec` layers into one bulk emission.

use crate::codegen::encode_plan::SeriesPlan;
use crate::codegen::planner::{PrimitiveListPlan, PrimitiveListPolicy};
use crate::codegen::type_registry::ScalarTransform;
use crate::ir::{PrimitiveLeaf, VecLayers};
use proc_macro2::TokenStream;
use quote::quote;

use super::emit::vec_emit_primitive;
use super::idents;
use super::leaf::{LeafArm, LeafArmKind, validity_into_option};
use super::leaf_kind::{
    BulkPrimitiveList, CapturedPrimitiveList, PrimitiveListCommon, PrimitiveListEncoding,
    ReplayedPrimitiveList, StreamPrimitiveList,
};
use super::{LeafCtx, leaf};

enum VecLeafSpec {
    Numeric {
        native: TokenStream,
        value_expr: TokenStream,
    },
    StringLike {
        value_expr: TokenStream,
        extra_decls: Vec<TokenStream>,
    },
    BinaryLike {
        value_expr: TokenStream,
    },
    Bool,
}

struct VecLeafPlan {
    spec: VecLeafSpec,
    leaf_dtype: TokenStream,
}

#[derive(Clone, Copy)]
struct VecLeafBuildCtx<'tokens, 'ir> {
    plan: PrimitiveListPolicy,
    ident_scope: idents::GeneratedIdentScope<'ir>,
    idx: usize,
    has_inner_option: bool,
    leaf_capacity_expr: &'tokens TokenStream,
    row_capacity: &'tokens syn::Ident,
    pa_root: &'tokens TokenStream,
}

fn bool_leaf_array_tokens(
    pa_root: &TokenStream,
    has_inner_option: bool,
    values_ident: &syn::Ident,
    validity_ident: &syn::Ident,
) -> TokenStream {
    if has_inner_option {
        let valid_opt = validity_into_option(validity_ident, pa_root);
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

/// The element-count expression that becomes the checked list offset for the
/// immediate schedule.
fn leaf_offsets_post_push_tokens(
    spec: &VecLeafSpec,
    plan: PrimitiveListPolicy,
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
) -> TokenStream {
    let flat = idents::vec_flat(idx);
    let view_buf = idents::vec_view_buf(idx);
    match spec {
        VecLeafSpec::Numeric { .. } => quote! { #flat.len() },
        VecLeafSpec::StringLike { .. } | VecLeafSpec::BinaryLike { .. } => {
            quote! { #view_buf.len() }
        }
        VecLeafSpec::Bool if plan.uses_exact_deferred_storage() => {
            let leaf_idx = idents::vec_leaf_idx(ident_scope, idx);
            quote! { #leaf_idx }
        }
        VecLeafSpec::Bool => {
            let values = idents::bool_values(idx);
            quote! { #values.len() }
        }
    }
}

fn leaf_prepare_segment_tokens(
    spec: &VecLeafSpec,
    plan: PrimitiveListPolicy,
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    has_inner_option: bool,
    row_capacity: &syn::Ident,
) -> TokenStream {
    let additional = idents::leaf_reserve_len();
    let validity_prepare = (has_inner_option
        && !(matches!(spec, VecLeafSpec::Numeric { .. })
            && matches!(plan, PrimitiveListPlan::StreamReserved(()))))
    .then(|| {
        let validity = idents::bool_validity(idx);
        quote! { #validity.extend_constant(#additional, true); }
    });
    match spec {
        VecLeafSpec::Numeric { .. } => {
            let values = idents::vec_flat(idx);
            let incremental_validity = (has_inner_option
                && matches!(plan, PrimitiveListPlan::StreamReserved(())))
            .then(|| {
                let validity = idents::bool_validity(idx);
                let growth = idents::vec_validity_growth(ident_scope, idx);
                quote! {
                    if #validity.len() < #values.len() + #additional {
                        let #growth = #validity
                            .len()
                            .max(#row_capacity)
                            .max(#additional);
                        #validity.extend_constant(
                            (#values.len() + #additional - #validity.len()).max(#growth),
                            true,
                        );
                    }
                }
            });
            quote! {
                #values.reserve(#additional);
                #validity_prepare
                #incremental_validity
            }
        }
        VecLeafSpec::StringLike { .. } | VecLeafSpec::BinaryLike { .. } => {
            let values = idents::vec_view_buf(idx);
            quote! {
                #values.reserve(#additional);
                #validity_prepare
            }
        }
        VecLeafSpec::Bool => {
            let values = idents::bool_values(idx);
            if has_inner_option {
                quote! {
                    #values.extend_constant(#additional, false);
                    #validity_prepare
                }
            } else {
                TokenStream::new()
            }
        }
    }
}

fn prepared_bitmap_set(
    helper: &syn::Ident,
    builder: &syn::Ident,
    index: &TokenStream,
    value: bool,
) -> TokenStream {
    quote! {
        // SAFETY: generated schedules prepare the complete bitmap range before
        // entering this element loop.
        unsafe { #helper(&mut #builder, #index, #value); }
    }
}

fn reserved_vec_push(helper: &syn::Ident, values: &syn::Ident, value: &TokenStream) -> TokenStream {
    let mapped = idents::leaf_value_mapped();
    quote! {
        let #mapped = #value;
        // SAFETY: generated schedules reserve this segment or allocate its
        // exact observed cardinality before entering the element loop.
        unsafe { #helper(&mut #values, #mapped); }
    }
}

fn build_vec_leaf_pieces(
    spec: &VecLeafSpec,
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> (TokenStream, TokenStream, TokenStream) {
    match spec {
        VecLeafSpec::Numeric { native, value_expr } => numeric_leaf_pieces(native, value_expr, ctx),
        VecLeafSpec::StringLike {
            value_expr,
            extra_decls,
        } => string_like_leaf_pieces(
            value_expr,
            extra_decls,
            ctx.ident_scope,
            ctx.idx,
            ctx.has_inner_option,
            ctx.leaf_capacity_expr,
            ctx.pa_root,
        ),
        VecLeafSpec::BinaryLike { value_expr } => binary_like_leaf_pieces(
            value_expr,
            ctx.ident_scope,
            ctx.idx,
            ctx.has_inner_option,
            ctx.leaf_capacity_expr,
            ctx.pa_root,
        ),
        VecLeafSpec::Bool => {
            if ctx.has_inner_option {
                bool_inner_option_leaf_pieces(
                    ctx.ident_scope,
                    ctx.idx,
                    ctx.leaf_capacity_expr,
                    ctx.pa_root,
                )
            } else {
                bool_bare_leaf_pieces(
                    ctx.ident_scope,
                    ctx.idx,
                    ctx.leaf_capacity_expr,
                    ctx.pa_root,
                    ctx.plan,
                )
            }
        }
    }
}

fn bool_bare_leaf_pieces(
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
    plan: PrimitiveListPolicy,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values(idx);
    let validity_ident = idents::bool_validity(idx);
    let v = idents::leaf_value();
    let (storage, push) = match plan {
        PrimitiveListPlan::BulkSegments(()) => (
            quote! {
                let mut #values_ident: ::std::vec::Vec<bool> =
                    ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
            },
            TokenStream::new(),
        ),
        PrimitiveListPlan::CaptureSegments(()) | PrimitiveListPlan::ReplayRows(()) => {
            let set_prepared_bitmap = idents::set_prepared_bitmap(ident_scope);
            let leaf_idx = idents::vec_leaf_idx(ident_scope, idx);
            let set_true = prepared_bitmap_set(
                &set_prepared_bitmap,
                &values_ident,
                &quote! { #leaf_idx },
                true,
            );
            (
                quote! {
                    let mut #values_ident: #pa_root::bitmap::MutableBitmap =
                        #pa_root::bitmap::MutableBitmap::from_len_zeroed(#leaf_capacity_expr);
                    let mut #leaf_idx: usize = 0;
                },
                quote! {
                    if *#v {
                        #set_true
                    }
                    #leaf_idx += 1;
                },
            )
        }
        PrimitiveListPlan::StreamReserved(()) => {
            unreachable!("bare Boolean leaves use append or deferred fill")
        }
    };
    let leaf_arr_inner = if matches!(plan, PrimitiveListPlan::BulkSegments(())) {
        quote! {
            #pa_root::array::BooleanArray::from_slice(&#values_ident)
        }
    } else {
        bool_leaf_array_tokens(pa_root, false, &values_ident, &validity_ident)
    };
    let leaf_arr = idents::leaf_arr();
    let leaf_arr_expr = quote! {
        let #leaf_arr: #pa_root::array::BooleanArray = #leaf_arr_inner;
    };
    (storage, push, leaf_arr_expr)
}

fn numeric_leaf_pieces(
    native: &TokenStream,
    value_expr: &TokenStream,
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> (TokenStream, TokenStream, TokenStream) {
    let VecLeafBuildCtx {
        plan,
        ident_scope,
        idx,
        has_inner_option,
        leaf_capacity_expr,
        row_capacity,
        pa_root,
    } = *ctx;
    let flat = idents::vec_flat(idx);
    let validity = idents::bool_validity(idx);
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    let incremental_validity =
        has_inner_option && matches!(plan, PrimitiveListPlan::StreamReserved(()));
    let value_capacity = if matches!(plan, PrimitiveListPlan::StreamReserved(())) {
        quote! { #row_capacity }
    } else {
        leaf_capacity_expr.clone()
    };
    let push_reserved = idents::push_reserved(ident_scope);
    let set_prepared_bitmap = idents::set_prepared_bitmap(ident_scope);
    let value_push = reserved_vec_push(&push_reserved, &flat, value_expr);
    let push = if has_inner_option {
        let default_value = quote! { <#native as ::std::default::Default>::default() };
        let default_push = reserved_vec_push(&push_reserved, &flat, &default_value);
        let null_index = quote! { #flat.len() - 1 };
        let set_null = prepared_bitmap_set(&set_prepared_bitmap, &validity, &null_index, false);
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #value_push
                }
                ::std::option::Option::None => {
                    #default_push
                    #set_null
                }
            }
        }
    } else {
        value_push
    };
    let storage = if incremental_validity {
        quote! {
            let mut #flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#value_capacity);
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(#row_capacity);
        }
    } else if has_inner_option {
        quote! {
            let mut #flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#value_capacity);
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::from_len_set(#leaf_capacity_expr);
        }
    } else {
        quote! {
            let mut #flat: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(#value_capacity);
        }
    };
    let leaf_arr_expr = if incremental_validity {
        let valid_opt = validity_into_option(&validity, pa_root);
        quote! {
            #validity.resize(#flat.len(), true);
            let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::new(
                    <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                    #flat.into(),
                    #valid_opt,
                );
        }
    } else if has_inner_option {
        let valid_opt = validity_into_option(&validity, pa_root);
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
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let view_buf = idents::vec_view_buf(idx);
    let validity = idents::bool_validity(idx);
    let set_prepared_bitmap = idents::set_prepared_bitmap(ident_scope);
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    for d in extra_decls {
        storage_parts.push(d.clone());
    }
    storage_parts.push(quote! {
        let mut #view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        let validity_decl = quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::from_len_set(#leaf_capacity_expr);
        };
        storage_parts.push(quote! {
            #validity_decl
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let push = if has_inner_option {
        let null_index = quote! { #view_buf.len() - 1 };
        let set_null = prepared_bitmap_set(&set_prepared_bitmap, &validity, &null_index, false);
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #view_buf.push_value_ignore_validity({ #value_expr });
                }
                ::std::option::Option::None => {
                    #view_buf.push_value_ignore_validity("");
                    #set_null
                }
            }
        }
    } else {
        quote! {
            #view_buf.push_value_ignore_validity({ #value_expr });
        }
    };
    let leaf_arr_expr = if has_inner_option {
        let valid_opt = validity_into_option(&validity, pa_root);
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

fn binary_like_leaf_pieces(
    value_expr: &TokenStream,
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let view_buf = idents::vec_view_buf(idx);
    let validity = idents::bool_validity(idx);
    let set_prepared_bitmap = idents::set_prepared_bitmap(ident_scope);
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    storage_parts.push(quote! {
        let mut #view_buf: #pa_root::array::MutableBinaryViewArray<[u8]> =
            #pa_root::array::MutableBinaryViewArray::<[u8]>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        let validity_decl = quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::from_len_set(#leaf_capacity_expr);
        };
        storage_parts.push(quote! {
            #validity_decl
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let empty = quote! { &[][..] };
    let push = if has_inner_option {
        let null_index = quote! { #view_buf.len() - 1 };
        let set_null = prepared_bitmap_set(&set_prepared_bitmap, &validity, &null_index, false);
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #view_buf.push_value_ignore_validity({ #value_expr });
                }
                ::std::option::Option::None => {
                    #view_buf.push_value_ignore_validity(#empty);
                    #set_null
                }
            }
        }
    } else {
        quote! {
            #view_buf.push_value_ignore_validity({ #value_expr });
        }
    };
    let leaf_arr_expr = if has_inner_option {
        let valid_opt = validity_into_option(&validity, pa_root);
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
    ident_scope: idents::GeneratedIdentScope<'_>,
    idx: usize,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values(idx);
    let validity_ident = idents::bool_validity(idx);
    let set_prepared_bitmap = idents::set_prepared_bitmap(ident_scope);
    let leaf_idx = idents::vec_leaf_idx(ident_scope, idx);
    let v = idents::leaf_value();
    let values_decl = quote! {
        let mut #values_ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::from_len_zeroed(#leaf_capacity_expr);
    };
    let validity_decl = quote! {
        let mut #validity_ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::from_len_set(#leaf_capacity_expr);
    };
    let storage = quote! {
        #values_decl
        #validity_decl
        let mut #leaf_idx: usize = 0;
    };
    let index = quote! { #leaf_idx };
    let value_true = prepared_bitmap_set(&set_prepared_bitmap, &values_ident, &index, true);
    let null = prepared_bitmap_set(&set_prepared_bitmap, &validity_ident, &index, false);
    let push = quote! {
        match #v {
            ::std::option::Option::Some(true) => {
                #value_true
            }
            ::std::option::Option::Some(false) => {}
            ::std::option::Option::None => {
                #null
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

fn vec_encoder(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecLayers,
    leaf_dtype: &TokenStream,
    plan: PrimitiveListPolicy,
) -> SeriesPlan {
    let encoding = lower_primitive_list(ctx, spec, shape, leaf_dtype, plan);
    vec_emit_primitive(&encoding, ctx.base.access, ctx.base.idx, shape, ctx.paths)
}

fn primitive_list_leaf_capacity(
    plan: PrimitiveListPolicy,
    shape: &VecLayers,
    row_capacity: &syn::Ident,
    shape_counts: &syn::Ident,
    leaf_count: &syn::Ident,
) -> TokenStream {
    match plan {
        PrimitiveListPlan::BulkSegments(()) => quote! { #row_capacity },
        PrimitiveListPlan::StreamReserved(()) => quote! { 0usize },
        PrimitiveListPlan::ReplayRows(()) => {
            let leaves = shape.depth();
            quote! { #shape_counts[#leaves] }
        }
        PrimitiveListPlan::CaptureSegments(()) => quote! { #leaf_count },
    }
}

fn lower_primitive_list(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecLayers,
    leaf_dtype: &TokenStream,
    plan: PrimitiveListPolicy,
) -> PrimitiveListEncoding {
    let pa_root = ctx.paths.polars_arrow_root();
    let row_capacity = ctx.base.row_capacity;
    let leaf_count = idents::vec_leaf_count(ctx.ident_scope, ctx.base.idx);
    let shape_counts = idents::vec_shape_counts(ctx.ident_scope, ctx.base.idx);
    let leaf_segments = idents::vec_leaf_segments(ctx.ident_scope, ctx.base.idx);
    let leaf_segment = idents::vec_leaf_segment(ctx.ident_scope, ctx.base.idx);
    let leaf_capacity_expr =
        primitive_list_leaf_capacity(plan, shape, row_capacity, &shape_counts, &leaf_count);
    let leaf_build_ctx = VecLeafBuildCtx {
        plan,
        ident_scope: ctx.ident_scope,
        idx: ctx.base.idx,
        has_inner_option: shape.has_inner_option(),
        leaf_capacity_expr: &leaf_capacity_expr,
        row_capacity,
        pa_root,
    };
    let (leaf_storage_decls, write_leaf, leaf_arr_expr) =
        build_vec_leaf_pieces(spec, &leaf_build_ctx);
    let common = PrimitiveListCommon {
        row_capacity: ctx.base.row_capacity.clone(),
        materialization: ctx.materialization.clone(),
        storage_decls: leaf_storage_decls,
        leaf_arr_expr,
        extra_imports: TokenStream::new(),
        leaf_logical_dtype: leaf_dtype.clone(),
    };
    match plan {
        PrimitiveListPlan::BulkSegments(()) => {
            let VecLeafSpec::Bool = spec else {
                unreachable!("only Boolean leaves support whole-segment writes")
            };
            let values = idents::bool_values(ctx.base.idx);
            let binding = idents::vec_leaf_segment(ctx.ident_scope, ctx.base.idx);
            let write = quote! {
                #values.extend(#binding.iter().copied());
            };
            PrimitiveListPlan::BulkSegments(BulkPrimitiveList {
                common,
                binding,
                write,
                leaf_offsets_post_push: leaf_offsets_post_push_tokens(
                    spec,
                    plan,
                    ctx.ident_scope,
                    ctx.base.idx,
                ),
            })
        }
        PrimitiveListPlan::StreamReserved(()) => {
            let prepare_segment = leaf_prepare_segment_tokens(
                spec,
                plan,
                ctx.ident_scope,
                ctx.base.idx,
                shape.has_inner_option(),
                ctx.base.row_capacity,
            );
            PrimitiveListPlan::StreamReserved(StreamPrimitiveList {
                common,
                prepare_segment,
                write_leaf,
                leaf_offsets_post_push: leaf_offsets_post_push_tokens(
                    spec,
                    plan,
                    ctx.ident_scope,
                    ctx.base.idx,
                ),
            })
        }
        PrimitiveListPlan::ReplayRows(()) => {
            let Some(row_replay) = ctx.row_replay else {
                unreachable!("direct replay ingredients must accompany a replay-row policy")
            };
            PrimitiveListPlan::ReplayRows(ReplayedPrimitiveList {
                common,
                shape_counts,
                leaf_offsets_post_push: leaf_offsets_post_push_tokens(
                    spec,
                    plan,
                    ctx.ident_scope,
                    ctx.base.idx,
                ),
                row: row_replay.row.clone(),
                replay: row_replay.replay.clone(),
                write_leaf,
            })
        }
        PrimitiveListPlan::CaptureSegments(()) => {
            PrimitiveListPlan::CaptureSegments(CapturedPrimitiveList {
                common,
                leaf_count,
                leaf_segments,
                leaf_segment,
                write_leaf,
            })
        }
    }
}

fn vec_encoder_bool_bare(
    ctx: &LeafCtx<'_>,
    shape: &VecLayers,
    plan: PrimitiveListPolicy,
) -> SeriesPlan {
    let leaf_dtype = PrimitiveLeaf::Bool.dtype(ctx.paths);
    vec_encoder(ctx, &VecLeafSpec::Bool, shape, &leaf_dtype, plan)
}

fn mapped_numeric_plan(
    ctx: &LeafCtx<'_>,
    leaf: ScalarTransform,
    native: TokenStream,
) -> VecLeafPlan {
    let v = idents::leaf_value();
    let mapped_v = crate::codegen::type_registry::map_primitive_expr(
        &quote! { #v },
        crate::codegen::type_registry::PrimitiveExprReceiver::Ref,
        leaf,
        ctx.decimal128_encode_trait,
        ctx.paths,
    );
    VecLeafPlan {
        spec: VecLeafSpec::Numeric {
            native,
            value_expr: mapped_v,
        },
        leaf_dtype: leaf.dtype(ctx.paths),
    }
}

fn vec_leaf_plan(leaf: PrimitiveLeaf<'_>, ctx: &LeafCtx<'_>) -> VecLeafPlan {
    let v = idents::leaf_value();
    match leaf {
        PrimitiveLeaf::Numeric(kind) => {
            let info = crate::codegen::type_registry::numeric_info_for(kind, ctx.paths);
            let value_expr = crate::codegen::type_registry::numeric_stored_value(
                kind,
                quote! { *#v },
                &info.native,
            );
            VecLeafPlan {
                spec: VecLeafSpec::Numeric {
                    native: info.native,
                    value_expr,
                },
                leaf_dtype: leaf.dtype(ctx.paths),
            }
        }
        PrimitiveLeaf::String => VecLeafPlan {
            spec: VecLeafSpec::StringLike {
                value_expr: quote! { #v.as_str() },
                extra_decls: Vec::new(),
            },
            leaf_dtype: leaf.dtype(ctx.paths),
        },
        PrimitiveLeaf::Binary => VecLeafPlan {
            spec: VecLeafSpec::BinaryLike {
                value_expr: quote! { ::core::convert::AsRef::<[u8]>::as_ref(#v) },
            },
            leaf_dtype: leaf.dtype(ctx.paths),
        },
        PrimitiveLeaf::Bool => VecLeafPlan {
            spec: VecLeafSpec::Bool,
            leaf_dtype: leaf.dtype(ctx.paths),
        },
        PrimitiveLeaf::DateTime(unit) => {
            mapped_numeric_plan(ctx, ScalarTransform::DateTime(unit), quote! { i64 })
        }
        PrimitiveLeaf::NaiveDateTime(unit) => {
            mapped_numeric_plan(ctx, ScalarTransform::NaiveDateTime(unit), quote! { i64 })
        }
        PrimitiveLeaf::NaiveTime => {
            mapped_numeric_plan(ctx, ScalarTransform::NaiveTime, quote! { i64 })
        }
        PrimitiveLeaf::Duration { unit, source } => mapped_numeric_plan(
            ctx,
            ScalarTransform::Duration { unit, source },
            quote! { i64 },
        ),
        PrimitiveLeaf::NaiveDate => {
            mapped_numeric_plan(ctx, ScalarTransform::NaiveDate, quote! { i32 })
        }
        PrimitiveLeaf::Decimal { precision, scale } => mapped_numeric_plan(
            ctx,
            ScalarTransform::Decimal { precision, scale },
            quote! { i128 },
        ),
        PrimitiveLeaf::AsString => {
            let scratch = idents::primitive_str_scratch(ctx.base.idx);
            let pp = ctx.paths.prelude();
            let value_expr = quote! {{
                use ::std::fmt::Write as _;
                #scratch.clear();
                ::std::write!(&mut #scratch, "{}", #v).map_err(|__df_fmt_err| {
                    #pp::polars_err!(
                        ComputeError:
                        "df-derive: as_string Display formatting failed: {}",
                        __df_fmt_err,
                    )
                })?;
                #scratch.as_str()
            }};
            VecLeafPlan {
                spec: VecLeafSpec::StringLike {
                    value_expr,
                    extra_decls: vec![quote! {
                        let mut #scratch: ::std::string::String =
                            ::std::string::String::new();
                    }],
                },
                leaf_dtype: leaf.dtype(ctx.paths),
            }
        }
        PrimitiveLeaf::AsStr(stringy) => {
            let value_expr = super::stringy_value_expr(
                stringy,
                &quote! { #v },
                super::StringyExprKind::MbvaValue,
            );
            VecLeafPlan {
                spec: VecLeafSpec::StringLike {
                    value_expr,
                    extra_decls: Vec::new(),
                },
                leaf_dtype: leaf.dtype(ctx.paths),
            }
        }
    }
}

pub(super) fn try_build_vec_encoder(
    leaf: PrimitiveLeaf<'_>,
    ctx: &LeafCtx<'_>,
    vec_shape: &VecLayers,
    list_plan: PrimitiveListPolicy,
) -> SeriesPlan {
    match leaf {
        PrimitiveLeaf::Bool => {
            if vec_shape.has_inner_option() {
                let plan = vec_leaf_plan(leaf, ctx);
                vec_encoder(ctx, &plan.spec, vec_shape, &plan.leaf_dtype, list_plan)
            } else {
                vec_encoder_bool_bare(ctx, vec_shape, list_plan)
            }
        }
        PrimitiveLeaf::Numeric(_)
        | PrimitiveLeaf::String
        | PrimitiveLeaf::Binary
        | PrimitiveLeaf::DateTime(_)
        | PrimitiveLeaf::NaiveDateTime(_)
        | PrimitiveLeaf::NaiveDate
        | PrimitiveLeaf::NaiveTime
        | PrimitiveLeaf::Duration { .. }
        | PrimitiveLeaf::Decimal { .. }
        | PrimitiveLeaf::AsString
        | PrimitiveLeaf::AsStr(_) => {
            let plan = vec_leaf_plan(leaf, ctx);
            vec_encoder(ctx, &plan.spec, vec_shape, &plan.leaf_dtype, list_plan)
        }
    }
}

pub(super) fn build_leaf(leaf: PrimitiveLeaf<'_>, ctx: &LeafCtx<'_>, kind: LeafArmKind) -> LeafArm {
    match leaf {
        PrimitiveLeaf::Numeric(num_kind) => leaf::numeric_leaf(ctx, num_kind, kind),
        PrimitiveLeaf::String => leaf::string_leaf(ctx, kind),
        PrimitiveLeaf::Bool => leaf::bool_leaf(ctx, kind),
        PrimitiveLeaf::Binary => leaf::binary_leaf(ctx, kind),
        PrimitiveLeaf::DateTime(unit) => leaf::datetime_leaf(ctx, unit, kind),
        PrimitiveLeaf::NaiveDateTime(unit) => leaf::naive_datetime_leaf(ctx, unit, kind),
        PrimitiveLeaf::NaiveDate => leaf::naive_date_leaf(ctx, kind),
        PrimitiveLeaf::NaiveTime => leaf::naive_time_leaf(ctx, kind),
        PrimitiveLeaf::Duration { unit, source } => leaf::duration_leaf(ctx, unit, source, kind),
        PrimitiveLeaf::Decimal { precision, scale } => {
            leaf::decimal_leaf(ctx, precision, scale, kind)
        }
        PrimitiveLeaf::AsString => leaf::as_string_leaf(ctx, kind),
        PrimitiveLeaf::AsStr(stringy) => leaf::as_str_leaf(ctx, stringy, kind),
    }
}
