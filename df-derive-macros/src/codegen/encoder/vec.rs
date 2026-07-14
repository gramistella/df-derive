//! Vec combinator and primitive leaf dispatch.
//!
//! `vec(inner)` fuses N consecutive `Vec` layers into one bulk emission.

use crate::codegen::type_registry::ScalarTransform;
use crate::ir::{PrimitiveLeaf, VecLayers};
use proc_macro2::TokenStream;
use quote::quote;

use super::emit::vec_emit_pep;
use super::idents;
use super::leaf::{LeafArm, LeafArmKind, validity_into_option};
use super::leaf_kind::PerElementPush;
use super::{Encoder, LeafCtx, leaf};

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

/// The element-count expression that becomes the checked list offset.
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

fn leaf_reserve_tokens(spec: &VecLeafSpec, idx: usize, has_inner_option: bool) -> TokenStream {
    let additional = idents::leaf_reserve_len();
    let validity_reserve = has_inner_option.then(|| {
        let validity = idents::bool_validity(idx);
        quote! { #validity.reserve(#additional); }
    });
    let values = match spec {
        VecLeafSpec::Numeric { .. } => idents::vec_flat(idx),
        VecLeafSpec::StringLike { .. } | VecLeafSpec::BinaryLike { .. } => {
            idents::vec_view_buf(idx)
        }
        VecLeafSpec::Bool => idents::bool_values(idx),
    };
    quote! {
        #values.reserve(#additional);
        #validity_reserve
    }
}

fn build_vec_leaf_pieces(
    spec: &VecLeafSpec,
    idx: usize,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    match spec {
        VecLeafSpec::Numeric { native, value_expr } => numeric_leaf_pieces(
            native,
            value_expr,
            idx,
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
            idx,
            has_inner_option,
            leaf_capacity_expr,
            pa_root,
        ),
        VecLeafSpec::BinaryLike { value_expr } => binary_like_leaf_pieces(
            value_expr,
            idx,
            has_inner_option,
            leaf_capacity_expr,
            pa_root,
        ),
        VecLeafSpec::Bool => {
            if has_inner_option {
                bool_inner_option_leaf_pieces(idx, leaf_capacity_expr, pa_root)
            } else {
                bool_bare_leaf_pieces(idx, leaf_capacity_expr, pa_root)
            }
        }
    }
}

fn bool_bare_leaf_pieces(
    idx: usize,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values(idx);
    let validity_ident = idents::bool_validity(idx);
    let v = idents::leaf_value();
    let values_decl = quote! {
        let mut #values_ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
    };
    let storage = quote! {
        #values_decl
    };
    let push = quote! {
        #values_ident.push(*#v);
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
    idx: usize,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let flat = idents::vec_flat(idx);
    let validity = idents::bool_validity(idx);
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    let storage = if has_inner_option {
        let validity_decl = quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
        };
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
    let push = if has_inner_option {
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #flat.push({ #value_expr });
                    #validity.push(true);
                }
                ::std::option::Option::None => {
                    #flat.push(<#native as ::std::default::Default>::default());
                    #validity.push(false);
                }
            }
        }
    } else {
        quote! {
            #flat.push(#value_expr);
        }
    };
    let leaf_arr_expr = if has_inner_option {
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
    idx: usize,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let view_buf = idents::vec_view_buf(idx);
    let validity = idents::bool_validity(idx);
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
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
        };
        storage_parts.push(quote! {
            #validity_decl
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let push = if has_inner_option {
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #view_buf.push_value_ignore_validity({ #value_expr });
                    #validity.push(true);
                }
                ::std::option::Option::None => {
                    #view_buf.push_value_ignore_validity("");
                    #validity.push(false);
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
    idx: usize,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let view_buf = idents::vec_view_buf(idx);
    let validity = idents::bool_validity(idx);
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
                #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
        };
        storage_parts.push(quote! {
            #validity_decl
        });
    }
    let storage = quote! { #(#storage_parts)* };
    let empty = quote! { &[][..] };
    let push = if has_inner_option {
        quote! {
            match #v {
                ::std::option::Option::Some(#v) => {
                    #view_buf.push_value_ignore_validity({ #value_expr });
                    #validity.push(true);
                }
                ::std::option::Option::None => {
                    #view_buf.push_value_ignore_validity(#empty);
                    #validity.push(false);
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
    idx: usize,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values(idx);
    let validity_ident = idents::bool_validity(idx);
    let v = idents::leaf_value();
    let values_decl = quote! {
        let mut #values_ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
    };
    let validity_decl = quote! {
        let mut #validity_ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(#leaf_capacity_expr);
    };
    let storage = quote! {
        #values_decl
        #validity_decl
    };
    let push = quote! {
        match #v {
            ::std::option::Option::Some(true) => {
                #values_ident.push(true);
                #validity_ident.push(true);
            }
            ::std::option::Option::Some(false) => {
                #values_ident.push(false);
                #validity_ident.push(true);
            }
            ::std::option::Option::None => {
                #values_ident.push(false);
                #validity_ident.push(false);
            }
        }
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
) -> Encoder {
    let pep = lower_to_pep(ctx, spec, shape, leaf_dtype);
    Encoder::Multi(vec_emit_pep(
        &pep,
        ctx.base.access,
        ctx.base.idx,
        shape,
        ctx.paths,
    ))
}

fn lower_to_pep(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecLayers,
    leaf_dtype: &TokenStream,
) -> PerElementPush {
    let pa_root = ctx.paths.polars_arrow_root();
    // A row count says nothing about the flattened element count of a list
    // column. Start conservatively and let the buffers grow during the one
    // traversal that actually observes the values.
    let leaf_capacity_expr = quote! { 0usize };
    let (leaf_storage_decls, per_elem_push, leaf_arr_expr) = build_vec_leaf_pieces(
        spec,
        ctx.base.idx,
        shape.has_inner_option(),
        &leaf_capacity_expr,
        pa_root,
    );
    let leaf_offsets_post_push = leaf_offsets_post_push_tokens(spec, ctx.base.idx);
    let reserve = leaf_reserve_tokens(spec, ctx.base.idx, shape.has_inner_option());
    PerElementPush {
        row_capacity: ctx.base.row_capacity.clone(),
        per_elem_push,
        reserve,
        storage_decls: leaf_storage_decls,
        leaf_arr_expr,
        leaf_offsets_post_push,
        extra_imports: TokenStream::new(),
        leaf_logical_dtype: leaf_dtype.clone(),
    }
}

fn vec_encoder_bool_bare(ctx: &LeafCtx<'_>, shape: &VecLayers) -> Encoder {
    let leaf_dtype = PrimitiveLeaf::Bool.dtype(ctx.paths);
    vec_encoder(ctx, &VecLeafSpec::Bool, shape, &leaf_dtype)
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
) -> Encoder {
    match leaf {
        PrimitiveLeaf::Bool => {
            if vec_shape.has_inner_option() {
                let plan = vec_leaf_plan(leaf, ctx);
                vec_encoder(ctx, &plan.spec, vec_shape, &plan.leaf_dtype)
            } else {
                vec_encoder_bool_bare(ctx, vec_shape)
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
            vec_encoder(ctx, &plan.spec, vec_shape, &plan.leaf_dtype)
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
