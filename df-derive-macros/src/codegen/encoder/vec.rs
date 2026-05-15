//! Vec combinator and primitive leaf dispatch.
//!
//! `vec(inner)` fuses N consecutive `Vec` layers into one bulk emission.

use crate::codegen::type_registry::ScalarTransform;
use crate::ir::{PrimitiveLeaf, VecLayers};
use proc_macro2::TokenStream;
use quote::quote;

use super::emit::vec_emit_pep;
use super::idents;
use super::leaf::{LeafArm, LeafArmKind, mb_decl_filled, validity_into_option};
use super::leaf_kind::PerElementPush;
use super::{Encoder, LeafCtx, leaf, list_offset_i64_expr};

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

fn build_vec_leaf_pieces(
    spec: &VecLeafSpec,
    has_inner_option: bool,
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    match spec {
        VecLeafSpec::Numeric { native, value_expr } => numeric_leaf_pieces(
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
                bool_inner_option_leaf_pieces(leaf_capacity_expr, pa_root)
            } else {
                bool_bare_leaf_pieces(leaf_capacity_expr, pa_root)
            }
        }
    }
}

fn bool_bare_leaf_pieces(
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let v = idents::leaf_value();
    let values_decl = mb_decl_filled(&values_ident, leaf_capacity_expr, false, pa_root);
    let storage = quote! {
        #values_decl
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
    let v = idents::leaf_value();
    let leaf_arr = idents::leaf_arr();
    let storage = if has_inner_option {
        let validity_decl = mb_decl_filled(&validity, leaf_capacity_expr, true, pa_root);
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
    for d in extra_decls {
        storage_parts.push(d.clone());
    }
    storage_parts.push(quote! {
        let mut #view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
    });
    if has_inner_option {
        let validity_decl = mb_decl_filled(&validity, leaf_capacity_expr, true, pa_root);
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
        let validity_decl = mb_decl_filled(&validity, leaf_capacity_expr, true, pa_root);
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
    leaf_capacity_expr: &TokenStream,
    pa_root: &TokenStream,
) -> (TokenStream, TokenStream, TokenStream) {
    let values_ident = idents::bool_values();
    let validity_ident = idents::bool_validity();
    let leaf_idx = idents::vec_leaf_idx();
    let v = idents::leaf_value();
    let values_decl = mb_decl_filled(&values_ident, leaf_capacity_expr, false, pa_root);
    let validity_decl = mb_decl_filled(&validity_ident, leaf_capacity_expr, true, pa_root);
    let storage = quote! {
        #values_decl
        #validity_decl
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

fn vec_encoder_series_local(idx: usize) -> syn::Ident {
    idents::vec_field_series(idx)
}

fn vec_encoder(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecLayers,
    leaf_dtype: &TokenStream,
) -> Encoder {
    let series_local = vec_encoder_series_local(ctx.base.idx);
    let pep = lower_to_pep(ctx, spec, shape, leaf_dtype);
    let decl = vec_emit_pep(&pep, ctx.base.access, ctx.base.idx, shape, ctx.paths);
    let name = ctx.base.name;
    let named = idents::field_named_series();
    let columns = idents::columns();
    let columnar = quote! {
        {
            #decl
            let #named = #series_local.with_name(#name.into());
            #columns.push(#named.into());
        }
    };
    Encoder::Multi { columnar }
}

fn lower_to_pep(
    ctx: &LeafCtx<'_>,
    spec: &VecLeafSpec,
    shape: &VecLayers,
    leaf_dtype: &TokenStream,
) -> PerElementPush {
    let pa_root = ctx.paths.polars_arrow_root();
    let total_leaves = idents::total_leaves();
    let leaf_capacity_expr = quote! { #total_leaves };
    let (leaf_storage_decls, per_elem_push, leaf_arr_expr) =
        build_vec_leaf_pieces(spec, shape.has_inner_option(), &leaf_capacity_expr, pa_root);
    let leaf_offsets_post_push = leaf_offsets_post_push_tokens(spec);
    PerElementPush {
        per_elem_push,
        storage_decls: leaf_storage_decls,
        leaf_arr_expr,
        leaf_offsets_post_push,
        extra_imports: TokenStream::new(),
        leaf_logical_dtype: leaf_dtype.clone(),
    }
}

pub(super) fn pep_for_primitive_leaf(
    leaf: PrimitiveLeaf<'_>,
    ctx: &LeafCtx<'_>,
    shape: &VecLayers,
) -> PerElementPush {
    let plan = vec_leaf_plan(leaf, ctx);
    lower_to_pep(ctx, &plan.spec, shape, &plan.leaf_dtype)
}

/// Bare-bool variant with a depth-1 `BooleanArray::from_slice` fast path.
fn vec_encoder_bool_bare(ctx: &LeafCtx<'_>, shape: &VecLayers) -> Encoder {
    // The depth-1 fast path uses `(&access).iter().copied()`, which
    // requires a plain `&bool`-yielding iterator. Any inner access chain
    // (Option or smart-pointer boundary) routes through the generalized
    // scanner so that boundary is resolved before the leaf push.
    if shape.depth() == 1 && !shape.any_outer_validity() && shape.inner_access.is_empty() {
        let pa_root = ctx.paths.polars_arrow_root();
        let pp = ctx.paths.prelude();
        let series_local = vec_encoder_series_local(ctx.base.idx);
        let leaf_dtype = PrimitiveLeaf::Bool.dtype(ctx.paths);
        let body = bool_bare_depth1_body(ctx.base.access, &leaf_dtype, pa_root, pp);
        let name = ctx.base.name;
        let named = idents::field_named_series();
        let columns = idents::columns();
        let decl = quote! { let #series_local: #pp::Series = { #body }; };
        let columnar = quote! {
            {
                #decl
                let #named = #series_local.with_name(#name.into());
                #columns.push(#named.into());
            }
        };
        return Encoder::Multi { columnar };
    }
    let leaf_dtype = PrimitiveLeaf::Bool.dtype(ctx.paths);
    vec_encoder(ctx, &VecLeafSpec::Bool, shape, &leaf_dtype)
}

fn bool_bare_depth1_body(
    access: &TokenStream,
    leaf_dtype: &TokenStream,
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
    let offset_ident = idents::list_offset();
    let offset = list_offset_i64_expr(&quote! { #flat.len() }, pp);
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
            let #offset_ident: i64 = #offset;
            #inner_offsets.push(#offset_ident);
        }
        let #leaf_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::from_slice(&#flat);
        let #offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            <#pa_root::offset::OffsetsBuffer<i64> as ::core::convert::TryFrom<::std::vec::Vec<i64>>>::try_from(#inner_offsets)?;
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
        )?
    }
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
