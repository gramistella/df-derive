use crate::codegen::MacroConfig;
use crate::ir::{AccessChain, VecLayers};
use proc_macro2::TokenStream;
use quote::quote;

use super::super::nested_columns::{
    NestedColumnIdents, NestedMaterializeBranches, NestedMaterializeKind, consume_nested_columns,
    inner_col_all_absent as nested_inner_col_all_absent,
    inner_col_direct as nested_inner_col_direct, inner_col_empty as nested_inner_col_empty,
    inner_col_take as nested_inner_col_take, nested_df_decl, nested_materialize_dispatch,
    nested_take_decl,
};
use super::super::shape_walk::{LayerIdents, ShapeEmitter, shape_assemble_list_stack};
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
    let df = idents::nested_df(field_idx);
    let take = idents::nested_take(field_idx);
    let columns = idents::columns();
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();
    let inner_chunk = idents::nested_inner_chunk();
    let inner_col = idents::nested_inner_col();
    let inner_rech = idents::nested_inner_rech();

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

    // Per-column inner-Series expressions for the four dispatch branches.
    let column_idents = NestedColumnIdents {
        df: &df,
        take: &take,
        col_name: &col_name,
        dtype: &dtype,
        inner_full: &inner_full,
    };
    let inner_col_direct = nested_inner_col_direct(column_idents);
    let inner_col_take = nested_inner_col_take(column_idents);
    let inner_col_empty = nested_inner_col_empty(&dtype, pp);
    let all_absent_len = quote! { #total_leaves };
    let inner_col_all_absent = nested_inner_col_all_absent(&dtype, &all_absent_len, pp);

    let series_direct = wrap_per_column_layers(
        &emitter,
        &inner_col_direct,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );
    let series_take = wrap_per_column_layers(
        &emitter,
        &inner_col_take,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );
    let series_empty = wrap_per_column_layers(
        &emitter,
        &inner_col_empty,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );
    let series_all_absent = wrap_per_column_layers(
        &emitter,
        &inner_col_all_absent,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );

    let consume_direct = consume_nested_columns(
        &columns,
        column_prefix,
        to_df_trait,
        type_path,
        &series_direct,
        pp,
    );
    let consume_take = consume_nested_columns(
        &columns,
        column_prefix,
        to_df_trait,
        type_path,
        &series_take,
        pp,
    );
    let consume_empty = consume_nested_columns(
        &columns,
        column_prefix,
        to_df_trait,
        type_path,
        &series_empty,
        pp,
    );
    let consume_all_absent = consume_nested_columns(
        &columns,
        column_prefix,
        to_df_trait,
        type_path,
        &series_all_absent,
        pp,
    );

    let df_decl = nested_df_decl(&df, type_path, columnar_trait, &flat);
    let take_decl = nested_take_decl(&take, &positions, pp);

    // Hoist offsets/validity freezes above the dispatch — collect-then-bulk
    // path reuses them per-column.
    let validity_freeze = emitter.freeze_validity_bitmaps();
    let offsets_freeze = emitter.freeze_offsets_buffers();

    let dispatch = nested_materialize_dispatch(
        NestedMaterializeKind::Vec { has_inner_option },
        &flat,
        Some(&total_leaves),
        &quote! { items.len() },
        NestedMaterializeBranches {
            validity_freeze,
            offsets_freeze,
            df_decl,
            take_decl,
            consume_direct,
            consume_take,
            consume_empty,
            consume_all_absent,
        },
    );

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

/// Wrap a per-column inner Series expression in the composed shape's
/// `LargeListArray::new` layers, routed through the assemble helper. Mirrors
/// `emit::ctb_layer_wrap` but owns its layer idents.
#[allow(clippy::too_many_arguments)]
fn wrap_per_column_layers(
    emitter: &ShapeEmitter<'_>,
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
    inner_chunk: &syn::Ident,
    inner_col: &syn::Ident,
    inner_rech: &syn::Ident,
    dtype: &syn::Ident,
) -> TokenStream {
    if emitter.shape.depth() == 0 {
        return inner_col_expr.clone();
    }
    let pa_root = emitter.pa_root;
    let chunk_decl = quote! {
        let #inner_col: #pp::Series = #inner_col_expr;
        let #inner_rech = #inner_col.rechunk();
        let #inner_chunk: #pp::ArrayRef = #inner_rech.chunks()[0].clone();
    };
    let wrap_layers = emitter.layer_wraps_clone();
    let stack = shape_assemble_list_stack(
        quote! { #inner_chunk },
        quote! { #inner_chunk.dtype().clone() },
        &wrap_layers,
        quote! { (*#dtype).clone() },
        pp,
        pa_root,
        &idents::tuple_layer_list_arr,
    );
    quote! {{
        #chunk_decl
        #stack
    }}
}
