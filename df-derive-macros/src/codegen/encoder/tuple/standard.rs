use crate::codegen::MacroConfig;
use crate::ir::WrapperShape;
use proc_macro2::TokenStream;
use quote::quote;

use super::super::{
    BaseCtx, Encoder, LeafCtx, NestedLeafCtx, build_encoder_with_option_receiver,
    build_nested_encoder, idents,
};
use super::{TupleLeafRoute, tuple_nested_type_path};

/// Emit one element column via the standard `build_encoder` /
/// `build_nested_encoder`, baking the resulting Leaf decls/push/series into
/// a single self-contained block when needed (the standard encoder's Leaf
/// shape is decls-before-loop + push-in-loop + series-after-loop; we
/// orchestrate that ourselves here so the caller's `columns.push(...)`
/// happens in the right order).
pub(super) fn emit_via_standard_encoder(
    access: &TokenStream,
    wrapper: &WrapperShape,
    leaf_route: TupleLeafRoute<'_>,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    emit_via_standard_encoder_with_option_receiver(
        access,
        wrapper,
        leaf_route,
        field_idx,
        column_prefix,
        config,
        None,
    )
}

pub(super) fn emit_via_standard_encoder_with_option_receiver(
    access: &TokenStream,
    wrapper: &WrapperShape,
    leaf_route: TupleLeafRoute<'_>,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
    option_some_receiver: Option<crate::codegen::type_registry::PrimitiveExprReceiver>,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    if let TupleLeafRoute::Nested(nested) = leaf_route {
        let nested_ty = tuple_nested_type_path(nested);
        let nested_ctx = NestedLeafCtx {
            base: BaseCtx {
                access,
                idx: field_idx,
                name: column_prefix,
            },
            ty: &nested_ty,
            columnar_trait: &config.traits.columnar,
            to_df_trait: &config.traits.to_dataframe,
            paths: &config.external_paths,
        };
        return build_nested_encoder(wrapper, &nested_ctx);
    }

    let TupleLeafRoute::Primitive(leaf) = leaf_route else {
        return TokenStream::new();
    };
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access,
            idx: field_idx,
            name: column_prefix,
        },
        decimal128_encode_trait: &config.traits.decimal128_encode,
        paths: &config.external_paths,
    };
    let enc = build_encoder_with_option_receiver(leaf, wrapper, &leaf_ctx, option_some_receiver);
    match enc {
        Encoder::Leaf {
            decls,
            push,
            series,
        } => {
            let it = idents::populator_iter();
            let named = idents::field_named_series();
            let series_local = idents::vec_field_series(field_idx);
            let columns = idents::columns();
            quote! {
                {
                    #(#decls)*
                    for #it in items { #push }
                    let #series_local: #pp::Series = #series;
                    let #named = #series_local.with_name(#column_prefix.into());
                    #columns.push(#named.into());
                }
            }
        }
        Encoder::Multi { columnar } => columnar,
    }
}
