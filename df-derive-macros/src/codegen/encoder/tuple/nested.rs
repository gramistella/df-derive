use crate::codegen::MacroConfig;
use crate::ir::{AccessChain, VecLayers};
use proc_macro2::TokenStream;
use quote::quote;

use super::super::nested_columns::{
    NestedMaterializeCtx, NestedWrapper, materialize_nested_columns,
};
use super::super::shape_walk::{LayerIdents, ShapeEmitter};
use super::super::{idents, idx_size_len_expr};
use super::projection::TupleProjection;
use super::vec_parent::{tuple_layer_counters, tuple_layer_idents, tuple_scan_leaf_body};

/// Emit a tuple element column for a nested-struct element under a
/// Vec-bearing parent. Walks the composed shape, collects `&Inner` refs at
/// the deepest binding (with projection applied), then dispatches the
/// inner type's `columnar_from_refs` and stacks `LargeListArray`s.
// Bench-sensitive generated-code builder: this intentionally keeps the
// composed-shape walk, nested dispatch, and list stacking adjacent so tuple
// Vec emission stays predictable while deeper factoring waits.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn emit_vec_parent_nested(
    parent_access: &TokenStream,
    composed_shape: &VecLayers,
    projection: TupleProjection<'_>,
    leaf_projection_access: Option<&AccessChain>,
    type_path: &TokenStream,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let columnar_trait = &config.traits.columnar;
    let to_df_trait = &config.traits.to_dataframe;
    let total_leaves = idents::nested_total(field_idx);
    let flat = idents::nested_flat(field_idx);
    let positions = idents::nested_positions(field_idx);

    let layers: Vec<LayerIdents> = (0..composed_shape.depth())
        .map(|i| tuple_layer_idents(field_idx, i))
        .collect();
    let layer_counters = tuple_layer_counters(field_idx, composed_shape.depth());
    let emitter = ShapeEmitter {
        shape: composed_shape,
        access: parent_access,
        layers: &layers,
        outer_some_prefix: idents::TUPLE_OUTER_SOME_PREFIX,
        precount_outer_some_prefix: idents::TUPLE_PRE_OUTER_SOME_PREFIX,
        total_counter: &total_leaves,
        layer_counters: &layer_counters,
        pp,
        pa_root,
        projection: projection.as_layer_projection(composed_shape),
    };
    let precount = emitter.precount();

    let has_inner_option = composed_shape.has_inner_option();

    // Per-element push body: collect &Inner refs (and positions on
    // inner-Option). The scan applies tuple projection before this body, so
    // the leaf binding is always `&Inner` or `Option<&Inner>`.
    let leaf_v = idents::leaf_value();
    let inner_v = idents::tuple_nested_inner_v();
    let projected_leaf = quote! { #leaf_v };
    let per_elem_push = if has_inner_option {
        let flat_idx = idx_size_len_expr(&flat, pp);
        quote! {
            match #projected_leaf {
                ::std::option::Option::Some(#inner_v) => {
                    #positions.push(::std::option::Option::Some(
                        #flat_idx,
                    ));
                    #flat.push(#inner_v);
                }
                ::std::option::Option::None => {
                    #positions.push(::std::option::Option::None);
                }
            }
        }
    } else {
        quote! {
            #flat.push(#projected_leaf);
        }
    };
    let leaf_offsets_post_push = if has_inner_option {
        quote! { #positions.len() }
    } else {
        quote! { #flat.len() }
    };

    let leaf_body = tuple_scan_leaf_body(
        composed_shape,
        projection,
        leaf_projection_access,
        &per_elem_push,
    );
    let scan = emitter.scan(&leaf_body, &leaf_offsets_post_push);
    let counter_for_depth = |layer: usize| -> TokenStream {
        let counter = &layer_counters[layer];
        quote! { #counter }
    };
    let offsets_decls = emitter.offsets_decls(&counter_for_depth);
    let validity_decls = emitter.validity_decls(&counter_for_depth);

    let positions_decl = if has_inner_option {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total_leaves);
        }
    } else {
        TokenStream::new()
    };

    let dispatch = materialize_nested_columns(&NestedMaterializeCtx {
        field_idx,
        ty: type_path,
        column_prefix,
        flat: &flat,
        positions: has_inner_option.then_some(&positions),
        total_len: quote! { #total_leaves },
        wrapper: NestedWrapper::List {
            shape: composed_shape,
            layers: &layers,
            arr_id_for_layer: idents::tuple_layer_list_arr,
        },
        columnar_trait,
        to_df_trait,
        paths: &config.external_paths,
    });

    quote! {
        {
            #precount
            let mut #flat: ::std::vec::Vec<&#type_path> =
                ::std::vec::Vec::with_capacity(#total_leaves);
            #positions_decl
            #offsets_decls
            #validity_decls
            #scan
            #dispatch
        }
    }
}
