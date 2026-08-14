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
use super::leaf::{LeafArm, LeafArmKind};
use super::leaf_kind::{
    BulkPrimitiveList, CapturedPrimitiveList, GroupedPrimitiveList, PrimitiveListCommon,
    PrimitiveListEncoding, StreamPrimitiveList,
};
use super::{LeafCtx, leaf};

enum VecLeafSpec {
    Numeric {
        native: TokenStream,
        value_expr: TokenStream,
        copy_identity: bool,
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
    shape: &'ir VecLayers,
    has_inner_option: bool,
    leaf_capacity_expr: &'tokens TokenStream,
    row_capacity: &'tokens syn::Ident,
    leaf_segment: &'tokens syn::Ident,
    leaf_segments: &'tokens syn::Ident,
    grouped_offsets: &'tokens syn::Ident,
    encode_support: &'tokens syn::Path,
    pa_root: &'tokens TokenStream,
    pp: &'tokens TokenStream,
}

struct LeafSegmentAccess {
    iteration_bind: syn::Ident,
    resolve_leaf: TokenStream,
}

impl LeafSegmentAccess {
    fn for_shape(shape: &VecLayers) -> Self {
        let leaf_bind = idents::leaf_value();
        if shape.inner_access.is_empty() || shape.inner_access.is_single_plain_option() {
            return Self {
                iteration_bind: leaf_bind,
                resolve_leaf: TokenStream::new(),
            };
        }

        let raw_bind = idents::leaf_value_raw();
        let chain_ref = super::access_chain_to_ref(&quote! { #raw_bind }, &shape.inner_access);
        let resolved = chain_ref.expr;
        let resolve_leaf = if chain_ref.has_option {
            quote! { let #leaf_bind: ::std::option::Option<_> = #resolved; }
        } else {
            quote! { let #leaf_bind = #resolved; }
        };
        Self {
            iteration_bind: raw_bind,
            resolve_leaf,
        }
    }

    fn mapper(&self, body: &TokenStream) -> TokenStream {
        let iteration_bind = &self.iteration_bind;
        let resolve_leaf = &self.resolve_leaf;
        quote! {
            |#iteration_bind| {
                #resolve_leaf
                #body
            }
        }
    }

    fn fallible_mapper(&self, body: &TokenStream, pp: &TokenStream) -> TokenStream {
        let iteration_bind = &self.iteration_bind;
        let resolve_leaf = &self.resolve_leaf;
        quote! {
            |#iteration_bind| -> #pp::PolarsResult<_> {
                #resolve_leaf
                #body
            }
        }
    }

    fn direct_loop(&self, segment: &syn::Ident, body: &TokenStream) -> TokenStream {
        let iteration_bind = &self.iteration_bind;
        let resolve_leaf = &self.resolve_leaf;
        quote! {
            for #iteration_bind in #segment.iter() {
                #resolve_leaf
                #body
            }
        }
    }
}

fn bool_leaf_array_tokens(
    pa_root: &TokenStream,
    has_inner_option: bool,
    values_ident: &syn::Ident,
    validity_ident: &syn::Ident,
    prepared_len: &syn::Ident,
    prepared_validity: &syn::Ident,
) -> TokenStream {
    if has_inner_option {
        quote! {
            {
                let #prepared_len = #values_ident.len();
                let #prepared_validity = #validity_ident.finish(#prepared_len);
                #pa_root::array::BooleanArray::new(
                    #pa_root::datatypes::ArrowDataType::Boolean,
                    #values_ident.finish(),
                    #prepared_validity,
                )
            }
        }
    } else {
        quote! {
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                #values_ident.finish(),
                ::std::option::Option::None,
            )
        }
    }
}

/// The element-count expression that becomes the checked list offset for the
/// immediate schedule.
fn leaf_offsets_post_push_tokens(spec: &VecLeafSpec, idx: usize) -> TokenStream {
    let flat = idents::vec_flat(idx);
    let view_buf = idents::vec_view_buf(idx);
    match spec {
        VecLeafSpec::Numeric { .. } => quote! { #flat.len() },
        VecLeafSpec::StringLike { .. } | VecLeafSpec::BinaryLike { .. } => {
            quote! { #view_buf.len() }
        }
        VecLeafSpec::Bool => {
            let values = idents::bool_values(idx);
            quote! { #values.len() }
        }
    }
}

fn build_vec_leaf_pieces(
    spec: &VecLeafSpec,
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> (TokenStream, TokenStream, TokenStream) {
    match spec {
        VecLeafSpec::Numeric {
            native,
            value_expr,
            copy_identity,
        } => numeric_leaf_pieces(native, value_expr, *copy_identity, ctx),
        VecLeafSpec::StringLike {
            value_expr,
            extra_decls,
        } => string_like_leaf_pieces(value_expr, extra_decls, ctx),
        VecLeafSpec::BinaryLike { value_expr } => binary_like_leaf_pieces(value_expr, ctx),
        VecLeafSpec::Bool => {
            if ctx.has_inner_option {
                bool_inner_option_leaf_pieces(ctx)
            } else {
                bool_bare_leaf_pieces(ctx)
            }
        }
    }
}

fn bool_bare_leaf_pieces(ctx: &VecLeafBuildCtx<'_, '_>) -> (TokenStream, TokenStream, TokenStream) {
    let VecLeafBuildCtx {
        plan,
        ident_scope,
        idx,
        shape,
        leaf_capacity_expr,
        leaf_segment,
        leaf_segments,
        grouped_offsets,
        encode_support,
        pa_root,
        ..
    } = *ctx;
    let values_ident = idents::bool_values(idx);
    let validity_ident = idents::bool_validity(idx);
    let prepared_len = idents::prepared_len(ident_scope, idx);
    let prepared_validity = idents::prepared_validity(ident_scope, idx);
    let v = idents::leaf_value();
    let (storage, fill_segment) = match plan {
        PrimitiveListPlan::BulkSegments(()) => (
            quote! {
                let mut #values_ident: ::std::vec::Vec<bool> =
                    ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
            },
            quote! {
                #values_ident.extend(#leaf_segment.iter().copied());
            },
        ),
        PrimitiveListPlan::CaptureSegments(()) => {
            let access = LeafSegmentAccess::for_shape(shape);
            let callback = access.mapper(&quote! { *#v });
            (
                quote! {
                    let mut #values_ident: #encode_support::PreparedBooleanValues =
                        #encode_support::PreparedBooleanValues::with_exact_len(
                            #leaf_capacity_expr,
                        );
                },
                quote! { #values_ident.extend_captured(&#leaf_segments, #callback); },
            )
        }
        PrimitiveListPlan::CaptureGroups(()) => {
            let access = LeafSegmentAccess::for_shape(shape);
            let callback = access.mapper(&quote! { *#v });
            (
                quote! {
                    let mut #values_ident: #encode_support::PreparedBooleanValues =
                        #encode_support::PreparedBooleanValues::with_exact_len(
                            #leaf_capacity_expr,
                        );
                },
                quote! {
                    let #grouped_offsets =
                        #values_ident.extend_grouped(&#leaf_segments, #callback);
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
        bool_leaf_array_tokens(
            pa_root,
            false,
            &values_ident,
            &validity_ident,
            &prepared_len,
            &prepared_validity,
        )
    };
    let leaf_arr = idents::leaf_arr();
    let leaf_arr_expr = quote! {
        let #leaf_arr: #pa_root::array::BooleanArray = #leaf_arr_inner;
    };
    (storage, fill_segment, leaf_arr_expr)
}

fn numeric_segment_fill(
    native: &TokenStream,
    value_expr: &TokenStream,
    copy_identity: bool,
    flat: &syn::Ident,
    validity: &syn::Ident,
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> TokenStream {
    let VecLeafBuildCtx {
        plan,
        shape,
        has_inner_option,
        leaf_segment,
        leaf_segments,
        grouped_offsets,
        pp,
        ..
    } = *ctx;
    let access = LeafSegmentAccess::for_shape(shape);
    let v = idents::leaf_value();
    let streamed = matches!(plan, PrimitiveListPlan::StreamReserved(()));

    if has_inner_option {
        if streamed {
            let optional_value = quote! {
                match #v {
                    ::std::option::Option::Some(#v) => {
                        ({ #value_expr }, true)
                    }
                    ::std::option::Option::None => {
                        (<#native as ::std::default::Default>::default(), false)
                    }
                }
            };
            let callback =
                access.fallible_mapper(&quote! { ::std::result::Result::Ok(#optional_value) }, pp);
            return quote! {
                #flat.try_extend_nullable_segment(
                    &mut #validity,
                    #leaf_segment,
                    #callback,
                )?;
            };
        }
        if copy_identity && shape.inner_access.is_single_plain_option() {
            if matches!(plan, PrimitiveListPlan::CaptureSegments(())) {
                return quote! {
                    #flat.extend_nullable_options_captured(
                        &mut #validity,
                        &#leaf_segments,
                    );
                };
            }
            if matches!(plan, PrimitiveListPlan::CaptureGroups(())) {
                return quote! {
                    let #grouped_offsets = #flat.extend_nullable_options_grouped(
                        &mut #validity,
                        &#leaf_segments,
                    );
                };
            }
        }
        let optional_value = quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    ::std::option::Option::Some({ #value_expr })
                }
                ::std::option::Option::None => {
                    ::std::option::Option::None
                }
            }
        };
        let callback = access.mapper(&optional_value);
        if matches!(plan, PrimitiveListPlan::CaptureSegments(())) {
            return quote! {
                #flat.extend_nullable_captured(
                    &mut #validity,
                    &#leaf_segments,
                    #callback,
                );
            };
        }
        if matches!(plan, PrimitiveListPlan::CaptureGroups(())) {
            return quote! {
                let #grouped_offsets = #flat.extend_nullable_grouped(
                    &mut #validity,
                    &#leaf_segments,
                    #callback,
                );
            };
        }
        return quote! {
            #flat.extend_nullable_segment(
                &mut #validity,
                #leaf_segment,
                #callback,
            );
        };
    }
    if streamed {
        let callback =
            access.fallible_mapper(&quote! { ::std::result::Result::Ok({ #value_expr }) }, pp);
        quote! { #flat.try_extend_segment(#leaf_segment, #callback)?; }
    } else {
        if copy_identity && matches!(plan, PrimitiveListPlan::CaptureSegments(())) {
            return quote! { #flat.extend_copied_captured(&#leaf_segments); };
        }
        if copy_identity && matches!(plan, PrimitiveListPlan::CaptureGroups(())) {
            return quote! {
                let #grouped_offsets = #flat.extend_copied_grouped(&#leaf_segments);
            };
        }
        let callback = access.mapper(value_expr);
        if matches!(plan, PrimitiveListPlan::CaptureSegments(())) {
            quote! { #flat.extend_captured(&#leaf_segments, #callback); }
        } else if matches!(plan, PrimitiveListPlan::CaptureGroups(())) {
            quote! {
                let #grouped_offsets = #flat.extend_grouped(&#leaf_segments, #callback);
            }
        } else {
            quote! { #flat.extend_segment(#leaf_segment, #callback); }
        }
    }
}

fn numeric_leaf_pieces(
    native: &TokenStream,
    value_expr: &TokenStream,
    copy_identity: bool,
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> (TokenStream, TokenStream, TokenStream) {
    let VecLeafBuildCtx {
        plan,
        ident_scope,
        idx,
        has_inner_option,
        leaf_capacity_expr,
        row_capacity,
        encode_support,
        pa_root,
        ..
    } = *ctx;
    let flat = idents::vec_flat(idx);
    let validity = idents::bool_validity(idx);
    let leaf_arr = idents::leaf_arr();
    let prepared_len = idents::prepared_len(ident_scope, idx);
    let prepared_validity = idents::prepared_validity(ident_scope, idx);
    let incremental_validity =
        has_inner_option && matches!(plan, PrimitiveListPlan::StreamReserved(()));
    let value_capacity = if matches!(plan, PrimitiveListPlan::StreamReserved(())) {
        quote! { #row_capacity }
    } else {
        leaf_capacity_expr.clone()
    };
    let storage = if incremental_validity {
        quote! {
            let mut #flat: #encode_support::ExactBuffer<#native> =
                #encode_support::ExactBuffer::with_capacity(#value_capacity);
            let mut #validity: #encode_support::PreparedValidity =
                #encode_support::PreparedValidity::with_capacity(#row_capacity);
        }
    } else if has_inner_option {
        quote! {
            let mut #flat: #encode_support::ExactBuffer<#native> =
                #encode_support::ExactBuffer::with_exact_len(#value_capacity);
            let mut #validity: #encode_support::PreparedValidity =
                #encode_support::PreparedValidity::with_exact_len(#leaf_capacity_expr);
        }
    } else if matches!(plan, PrimitiveListPlan::StreamReserved(())) {
        quote! {
            let mut #flat: #encode_support::ExactBuffer<#native> =
                #encode_support::ExactBuffer::with_capacity(#value_capacity);
        }
    } else {
        quote! {
            let mut #flat: #encode_support::ExactBuffer<#native> =
                #encode_support::ExactBuffer::with_exact_len(#value_capacity);
        }
    };
    let fill_segment =
        numeric_segment_fill(native, value_expr, copy_identity, &flat, &validity, ctx);
    let leaf_arr_expr = if has_inner_option {
        quote! {
            let #prepared_len = #flat.len();
            let #prepared_validity = #validity.finish(#prepared_len);
            let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::new(
                    <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                    #flat.finish().into(),
                    #prepared_validity,
                );
        }
    } else {
        quote! {
            let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                #pa_root::array::PrimitiveArray::<#native>::from_vec(#flat.finish());
        }
    };
    (storage, fill_segment, leaf_arr_expr)
}

fn view_segment_fill(
    value_expr: &TokenStream,
    null_value: &TokenStream,
    view_buf: &syn::Ident,
    validity: &syn::Ident,
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> TokenStream {
    let VecLeafBuildCtx {
        plan,
        shape,
        has_inner_option,
        leaf_segment,
        leaf_segments,
        grouped_offsets,
        pp,
        ..
    } = *ctx;
    let access = LeafSegmentAccess::for_shape(shape);
    let v = idents::leaf_value();
    let streamed = matches!(plan, PrimitiveListPlan::StreamReserved(()));

    if has_inner_option {
        let write_and_validity = quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #view_buf.push_value_ignore_validity({ #value_expr });
                    true
                }
                ::std::option::Option::None => {
                    #view_buf.push_value_ignore_validity(#null_value);
                    false
                }
            }
        };
        if streamed {
            let callback = access.fallible_mapper(
                &quote! { ::std::result::Result::Ok(#write_and_validity) },
                pp,
            );
            return quote! {
                #validity.try_extend_segment(
                    #view_buf.len(),
                    #leaf_segment,
                    #callback,
                )?;
            };
        }
        let callback = access.mapper(&write_and_validity);
        if matches!(plan, PrimitiveListPlan::CaptureSegments(())) {
            return quote! {
                #validity.extend_captured(
                    #view_buf.len(),
                    &#leaf_segments,
                    #callback,
                );
            };
        }
        if matches!(plan, PrimitiveListPlan::CaptureGroups(())) {
            return quote! {
                let #grouped_offsets = #validity.extend_grouped(
                    #view_buf.len(),
                    &#leaf_segments,
                    #callback,
                );
            };
        }
        return quote! {
            #validity.extend_segment(
                #view_buf.len(),
                #leaf_segment,
                #callback,
            );
        };
    }

    let push = quote! { #view_buf.push_value_ignore_validity({ #value_expr }); };
    if matches!(plan, PrimitiveListPlan::CaptureSegments(())) {
        let callback = access.mapper(&push);
        return quote! { #leaf_segments.visit(#callback); };
    }
    if matches!(plan, PrimitiveListPlan::CaptureGroups(())) {
        let loop_body = access.direct_loop(leaf_segment, &push);
        return quote! {
            let #grouped_offsets = #leaf_segments.visit_segments(|#leaf_segment| {
                #loop_body
            });
        };
    }
    let loop_body = access.direct_loop(leaf_segment, &push);
    let reserve = streamed.then(|| quote! { #view_buf.reserve(#leaf_segment.len()); });
    quote! {
        #reserve
        #loop_body
    }
}

fn string_like_leaf_pieces(
    value_expr: &TokenStream,
    extra_decls: &[TokenStream],
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> (TokenStream, TokenStream, TokenStream) {
    let VecLeafBuildCtx {
        plan,
        ident_scope,
        idx,
        has_inner_option,
        leaf_capacity_expr,
        encode_support,
        pa_root,
        ..
    } = *ctx;
    let view_buf = idents::vec_view_buf(idx);
    let validity = idents::bool_validity(idx);
    let leaf_arr = idents::leaf_arr();
    let prepared_validity = idents::prepared_validity(ident_scope, idx);
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    for d in extra_decls {
        storage_parts.push(d.clone());
    }
    storage_parts.push(quote! {
        let mut #view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        let validity_decl = if matches!(plan, PrimitiveListPlan::StreamReserved(())) {
            quote! {
                let mut #validity: #encode_support::PreparedValidity =
                    #encode_support::PreparedValidity::with_capacity(#leaf_capacity_expr);
            }
        } else {
            quote! {
                let mut #validity: #encode_support::PreparedValidity =
                    #encode_support::PreparedValidity::with_exact_len(#leaf_capacity_expr);
            }
        };
        storage_parts.push(quote! {
            #validity_decl
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let fill_segment = view_segment_fill(value_expr, &quote! { "" }, &view_buf, &validity, ctx);
    let leaf_arr_expr = if has_inner_option {
        quote! {
            let #prepared_validity = #validity.finish(#view_buf.len());
            let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf
                .freeze()
                .with_validity(#prepared_validity);
        }
    } else {
        quote! {
            let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf.freeze();
        }
    };
    (storage, fill_segment, leaf_arr_expr)
}

fn binary_like_leaf_pieces(
    value_expr: &TokenStream,
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> (TokenStream, TokenStream, TokenStream) {
    let VecLeafBuildCtx {
        plan,
        ident_scope,
        idx,
        has_inner_option,
        leaf_capacity_expr,
        encode_support,
        pa_root,
        ..
    } = *ctx;
    let view_buf = idents::vec_view_buf(idx);
    let validity = idents::bool_validity(idx);
    let leaf_arr = idents::leaf_arr();
    let prepared_validity = idents::prepared_validity(ident_scope, idx);
    let mut storage_parts: Vec<TokenStream> = Vec::new();
    storage_parts.push(quote! {
        let mut #view_buf: #pa_root::array::MutableBinaryViewArray<[u8]> =
            #pa_root::array::MutableBinaryViewArray::<[u8]>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        let validity_decl = if matches!(plan, PrimitiveListPlan::StreamReserved(())) {
            quote! {
                let mut #validity: #encode_support::PreparedValidity =
                    #encode_support::PreparedValidity::with_capacity(#leaf_capacity_expr);
            }
        } else {
            quote! {
                let mut #validity: #encode_support::PreparedValidity =
                    #encode_support::PreparedValidity::with_exact_len(#leaf_capacity_expr);
            }
        };
        storage_parts.push(quote! {
            #validity_decl
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let fill_segment =
        view_segment_fill(value_expr, &quote! { &[][..] }, &view_buf, &validity, ctx);
    let leaf_arr_expr = if has_inner_option {
        quote! {
            let #prepared_validity = #validity.finish(#view_buf.len());
            let #leaf_arr: #pa_root::array::BinaryViewArray = #view_buf
                .freeze()
                .with_validity(#prepared_validity);
        }
    } else {
        quote! {
            let #leaf_arr: #pa_root::array::BinaryViewArray = #view_buf.freeze();
        }
    };
    (storage, fill_segment, leaf_arr_expr)
}

fn bool_inner_option_leaf_pieces(
    ctx: &VecLeafBuildCtx<'_, '_>,
) -> (TokenStream, TokenStream, TokenStream) {
    let VecLeafBuildCtx {
        ident_scope,
        idx,
        shape,
        leaf_capacity_expr,
        leaf_segment,
        leaf_segments,
        grouped_offsets,
        encode_support,
        pa_root,
        ..
    } = *ctx;
    let values_ident = idents::bool_values(idx);
    let validity_ident = idents::bool_validity(idx);
    let v = idents::leaf_value();
    let prepared_len = idents::prepared_len(ident_scope, idx);
    let prepared_validity = idents::prepared_validity(ident_scope, idx);
    let values_decl = quote! {
        let mut #values_ident: #encode_support::PreparedBooleanValues =
            #encode_support::PreparedBooleanValues::with_exact_len(#leaf_capacity_expr);
    };
    let validity_decl = quote! {
        let mut #validity_ident: #encode_support::PreparedValidity =
            #encode_support::PreparedValidity::with_exact_len(#leaf_capacity_expr);
    };
    let storage = quote! {
        #values_decl
        #validity_decl
    };
    let optional_value = quote! {
        match #v {
            ::std::option::Option::Some(#v) => ::std::option::Option::Some(*#v),
            ::std::option::Option::None => ::std::option::Option::None,
        }
    };
    let callback = LeafSegmentAccess::for_shape(shape).mapper(&optional_value);
    let fill_segment = if matches!(ctx.plan, PrimitiveListPlan::CaptureSegments(())) {
        quote! {
            #values_ident.extend_nullable_captured(
                &mut #validity_ident,
                &#leaf_segments,
                #callback,
            );
        }
    } else if matches!(ctx.plan, PrimitiveListPlan::CaptureGroups(())) {
        quote! {
            let #grouped_offsets = #values_ident.extend_nullable_grouped(
                &mut #validity_ident,
                &#leaf_segments,
                #callback,
            );
        }
    } else {
        quote! {
            #values_ident.extend_nullable_segment(
                &mut #validity_ident,
                #leaf_segment,
                #callback,
            );
        }
    };
    let leaf_arr_inner = bool_leaf_array_tokens(
        pa_root,
        true,
        &values_ident,
        &validity_ident,
        &prepared_len,
        &prepared_validity,
    );
    let leaf_arr = idents::leaf_arr();
    let leaf_arr_expr = quote! {
        let #leaf_arr: #pa_root::array::BooleanArray = #leaf_arr_inner;
    };
    (storage, fill_segment, leaf_arr_expr)
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
    row_capacity: &syn::Ident,
    leaf_count: &syn::Ident,
    leaf_segments: &syn::Ident,
) -> TokenStream {
    match plan {
        PrimitiveListPlan::BulkSegments(()) => quote! { #row_capacity },
        PrimitiveListPlan::StreamReserved(()) => quote! { 0usize },
        PrimitiveListPlan::CaptureSegments(()) => quote! { #leaf_count },
        PrimitiveListPlan::CaptureGroups(()) => quote! { #leaf_segments.len() },
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
    let leaf_segments = idents::vec_leaf_segments(ctx.ident_scope, ctx.base.idx);
    let leaf_segment = idents::vec_leaf_segment(ctx.ident_scope, ctx.base.idx);
    let grouped_offsets = idents::vec_layer_offsets(ctx.base.idx, shape.depth().saturating_sub(1));
    let leaf_capacity_expr =
        primitive_list_leaf_capacity(plan, row_capacity, &leaf_count, &leaf_segments);
    let leaf_build_ctx = VecLeafBuildCtx {
        plan,
        ident_scope: ctx.ident_scope,
        idx: ctx.base.idx,
        shape,
        has_inner_option: shape.has_inner_option(),
        leaf_capacity_expr: &leaf_capacity_expr,
        row_capacity,
        leaf_segment: &leaf_segment,
        leaf_segments: &leaf_segments,
        grouped_offsets: &grouped_offsets,
        encode_support: ctx.encode_support,
        pa_root,
        pp: ctx.paths.prelude(),
    };
    let (leaf_storage_decls, fill_segment, leaf_arr_expr) =
        build_vec_leaf_pieces(spec, &leaf_build_ctx);
    let common = PrimitiveListCommon {
        row_capacity: ctx.base.row_capacity.clone(),
        materialization: ctx.materialization.clone(),
        storage_decls: leaf_storage_decls,
        leaf_arr_expr,
        leaf_segment: leaf_segment.clone(),
        extra_imports: TokenStream::new(),
        leaf_logical_dtype: leaf_dtype.clone(),
    };
    match plan {
        PrimitiveListPlan::BulkSegments(()) => {
            let VecLeafSpec::Bool = spec else {
                unreachable!("only Boolean leaves support whole-segment writes")
            };
            PrimitiveListPlan::BulkSegments(BulkPrimitiveList {
                common,
                fill_segment,
                leaf_offsets_post_fill: leaf_offsets_post_push_tokens(spec, ctx.base.idx),
            })
        }
        PrimitiveListPlan::StreamReserved(()) => {
            PrimitiveListPlan::StreamReserved(StreamPrimitiveList {
                common,
                fill_segment,
                leaf_offsets_post_fill: leaf_offsets_post_push_tokens(spec, ctx.base.idx),
            })
        }
        PrimitiveListPlan::CaptureSegments(()) => {
            PrimitiveListPlan::CaptureSegments(CapturedPrimitiveList {
                common,
                encode_support: ctx.encode_support.clone(),
                leaf_count,
                leaf_segments,
                fill_segment,
            })
        }
        PrimitiveListPlan::CaptureGroups(()) => {
            PrimitiveListPlan::CaptureGroups(GroupedPrimitiveList {
                common,
                encode_support: ctx.encode_support.clone(),
                leaf_groups: leaf_segments,
                fill_group: fill_segment,
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
            copy_identity: false,
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
                    copy_identity: !kind.is_nonzero() && !kind.is_widened(),
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
