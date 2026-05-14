use crate::codegen::MacroConfig;
use crate::ir::{AccessChain, VecLayers};
use proc_macro2::TokenStream;
use quote::quote;

use super::super::shape_walk::{
    LayerIdents, freeze_offsets_buf, freeze_validity_bitmap, shape_assemble_list_stack,
};
use super::super::{idents, idx_size_len_expr};
use super::projection::TupleProjection;
use super::vec_parent::{
    TupleScanLeaf, build_offsets_decls, build_precount, build_scan, build_validity_decls,
    tuple_layer_counters, tuple_layer_idents, tuple_layer_wraps_clone,
};

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
    let columnar_trait = &config.columnar_trait_path;
    let to_df_trait = &config.to_dataframe_trait_path;
    let total_leaves = idents::nested_total(field_idx);
    let flat = idents::nested_flat(field_idx);
    let positions = idents::nested_positions(field_idx);
    let df = idents::nested_df(field_idx);
    let take = idents::nested_take(field_idx);
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let prefixed = idents::nested_prefixed_name();
    let inner_series = idents::nested_inner_series();
    let inner_full = idents::nested_inner_full();
    let inner_chunk = idents::nested_inner_chunk();
    let inner_col = idents::nested_inner_col();
    let inner_rech = idents::nested_inner_rech();
    let named = idents::field_named_series();
    let validate_nested_frame = idents::validate_nested_frame();
    let validate_nested_column_dtype = idents::validate_nested_column_dtype();

    let layers: Vec<LayerIdents> = (0..composed_shape.depth())
        .map(|i| tuple_layer_idents(field_idx, i))
        .collect();
    let layer_counters = tuple_layer_counters(field_idx, composed_shape.depth());

    let precount = build_precount(
        composed_shape,
        &layers,
        &layer_counters,
        &total_leaves,
        parent_access,
        projection,
    );

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

    let scan = build_scan(
        composed_shape,
        &layers,
        parent_access,
        projection,
        TupleScanLeaf {
            projection_access: leaf_projection_access,
            per_elem_push: &per_elem_push,
            offsets_post_push: &leaf_offsets_post_push,
            pp,
        },
    );
    let offsets_decls = build_offsets_decls(&layers, &layer_counters);
    let validity_decls = build_validity_decls(composed_shape, &layers, &layer_counters, pa_root);

    let positions_decl = if has_inner_option {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total_leaves);
        }
    } else {
        TokenStream::new()
    };

    // Per-column inner-Series expressions for the four dispatch branches.
    let inner_col_direct = quote! {{
        let #inner_full = #df.column(#col_name)?.as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
        #inner_full.clone()
    }};
    let inner_col_take = quote! {{
        let #inner_full = #df.column(#col_name)?.as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
        #inner_full.take(&#take)?
    }};
    let inner_col_empty = quote! { #pp::Series::new_empty("".into(), #dtype) };
    let inner_col_all_absent = quote! {
        #pp::Series::new_empty("".into(), #dtype)
            .extend_constant(#pp::AnyValue::Null, #total_leaves)?
    };

    let series_direct = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_direct,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
    );
    let series_take = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_take,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
    );
    let series_empty = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_empty,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
    );
    let series_all_absent = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_all_absent,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
    );

    let consume = |series_expr: &TokenStream| -> TokenStream {
        quote! {
            for (#col_name, #dtype) in <#type_path as #to_df_trait>::schema()? {
                let #col_name: &str = #col_name.as_str();
                let #dtype: &#pp::DataType = &#dtype;
                {
                    let #prefixed = ::std::format!("{}.{}", #column_prefix, #col_name);
                    let #inner_series: #pp::Series = #series_expr;
                    let #named = #inner_series.with_name(#prefixed.as_str().into());
                    columns.push(#named.into());
                }
            }
        }
    };

    let consume_direct = consume(&series_direct);
    let consume_take = consume(&series_take);
    let consume_empty = consume(&series_empty);
    let consume_all_absent = consume(&series_all_absent);

    let df_decl = quote! {
        let #df = <#type_path as #columnar_trait>::columnar_from_refs(&#flat)?;
        #validate_nested_frame(&#df, #flat.len(), ::core::any::type_name::<#type_path>())?;
    };
    let take_decl = quote! {
        let #take: #pp::IdxCa =
            <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                "".into(),
                #positions.iter().copied(),
            );
    };

    // Hoist offsets/validity freezes above the dispatch — collect-then-bulk
    // path reuses them per-column.
    let mut validity_freezes: Vec<TokenStream> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if !composed_shape.layers[i].has_outer_validity() {
            continue;
        }
        validity_freezes.push(freeze_validity_bitmap(
            &layer.validity_bm,
            &layer.validity_mb,
            pa_root,
        ));
    }
    let mut offsets_freezes: Vec<TokenStream> = Vec::new();
    for layer in &layers {
        offsets_freezes.push(freeze_offsets_buf(
            &layer.offsets_buf,
            &layer.offsets,
            pa_root,
        ));
    }
    let validity_freeze = quote! { #(#validity_freezes)* };
    let offsets_freeze = quote! { #(#offsets_freezes)* };

    let dispatch = if has_inner_option {
        quote! {
            #validity_freeze
            if #total_leaves == 0 {
                #offsets_freeze
                #consume_empty
            } else if #flat.is_empty() {
                #offsets_freeze
                #consume_all_absent
            } else if #flat.len() == #total_leaves {
                #df_decl
                #offsets_freeze
                #consume_direct
            } else {
                #df_decl
                #take_decl
                #offsets_freeze
                #consume_take
            }
        }
    } else {
        quote! {
            #validity_freeze
            if #flat.is_empty() {
                #offsets_freeze
                #consume_empty
            } else {
                #df_decl
                #offsets_freeze
                #consume_direct
            }
        }
    };

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
    shape: &VecLayers,
    layers: &[LayerIdents],
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
    inner_chunk: &syn::Ident,
    inner_col: &syn::Ident,
    inner_rech: &syn::Ident,
    dtype: &syn::Ident,
    pa_root: &TokenStream,
) -> TokenStream {
    if shape.depth() == 0 {
        return inner_col_expr.clone();
    }
    let chunk_decl = quote! {
        let #inner_col: #pp::Series = #inner_col_expr;
        let #inner_rech = #inner_col.rechunk();
        let #inner_chunk: #pp::ArrayRef = #inner_rech.chunks()[0].clone();
    };
    let wrap_layers = tuple_layer_wraps_clone(shape, layers);
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
