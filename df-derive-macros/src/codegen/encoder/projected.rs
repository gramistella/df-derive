use crate::codegen::MacroConfig;
use crate::ir::{AccessChain, ColumnIR, ColumnSource, ProjectionContext, VecLayers, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents::{self, LayerIdents};
use super::nested_columns::{NestedMaterializeCtx, NestedWrapper, materialize_nested_columns};
use super::shape_walk::{LayerProjection, ShapeEmitter, shape_assemble_list_stack};
use super::{BaseCtx, LeafCtx, access_chain_to_option_ref, access_chain_to_ref, idx_size_len_expr};

#[derive(Clone, Copy)]
struct ParentVecProjection<'a> {
    projection: LayerProjection<'a>,
    parent_inner_access: &'a AccessChain,
}

pub(in crate::codegen) fn build_projected_vec_primitive(
    column: &ColumnIR,
    leaf: crate::ir::PrimitiveLeaf<'_>,
    idx: usize,
    config: &MacroConfig,
) -> TokenStream {
    let WrapperShape::Vec(shape) = &column.wrapper_shape else {
        return TokenStream::new();
    };
    let parent_access = projected_parent_access(column);
    let projection_path = projected_path_tokens(column);
    let projection = parent_vec_projection(column, &projection_path);
    emit_projected_vec_primitive(
        &parent_access,
        shape,
        projection,
        leaf,
        idx,
        &column.name,
        config,
    )
}

pub(in crate::codegen) fn build_projected_vec_nested(
    column: &ColumnIR,
    type_path: &TokenStream,
    idx: usize,
    config: &MacroConfig,
) -> TokenStream {
    let WrapperShape::Vec(shape) = &column.wrapper_shape else {
        return TokenStream::new();
    };
    let parent_access = projected_parent_access(column);
    let projection_path = projected_path_tokens(column);
    let projection = parent_vec_projection(column, &projection_path);
    emit_projected_vec_nested(
        &parent_access,
        shape,
        projection,
        type_path,
        idx,
        &column.name,
        config,
    )
}

fn projected_parent_access(column: &ColumnIR) -> TokenStream {
    let ColumnSource::TupleProjection {
        root,
        context: ProjectionContext::ParentVec { .. },
        ..
    } = &column.source
    else {
        return TokenStream::new();
    };
    let it = idents::populator_iter();
    crate::codegen::access::field_source_access(root, &it)
}

fn projected_path_tokens(column: &ColumnIR) -> TokenStream {
    let ColumnSource::TupleProjection {
        path,
        context: ProjectionContext::ParentVec { .. },
        ..
    } = &column.source
    else {
        return TokenStream::new();
    };
    crate::codegen::access::projection_path_suffix(path)
}

fn parent_vec_projection<'a>(
    column: &'a ColumnIR,
    path_tokens: &'a TokenStream,
) -> ParentVecProjection<'a> {
    let ColumnSource::TupleProjection {
        path,
        context:
            ProjectionContext::ParentVec {
                projection_layer,
                parent_inner_access,
            },
        ..
    } = &column.source
    else {
        unreachable!("projected vec emit requires a ParentVec projection source");
    };
    ParentVecProjection {
        projection: LayerProjection {
            layer: *projection_layer,
            path: path_tokens,
            parent_access: parent_inner_access,
            smart_ptr_depth: crate::codegen::access::projection_terminal_smart_ptr_depth(path),
        },
        parent_inner_access,
    }
}

fn leaf_projection_access(
    shape: &VecLayers,
    projection: &ParentVecProjection<'_>,
) -> Option<AccessChain> {
    if projection.projection.layer != shape.depth() {
        return None;
    }
    let prefix_len = projection.parent_inner_access.steps.len();
    debug_assert!(
        shape
            .inner_access
            .steps
            .starts_with(&projection.parent_inner_access.steps),
        "projected tuple leaf access must include the parent inner access prefix"
    );
    Some(AccessChain {
        steps: shape.inner_access.steps[prefix_len..].to_vec(),
    })
}

fn project_tuple_element_ref(
    tuple_ref: &TokenStream,
    projection: &LayerProjection<'_>,
) -> TokenStream {
    let path = projection.path;
    let mut projected = quote! { (*(#tuple_ref)) #path };
    for _ in 0..projection.smart_ptr_depth {
        projected = quote! { (*(#projected)) };
    }
    quote! { &(#projected) }
}

fn apply_element_access(
    projected_ref: &TokenStream,
    element_access: &AccessChain,
) -> (TokenStream, bool) {
    if element_access.option_layers() > 0 {
        return (
            access_chain_to_option_ref(projected_ref, element_access),
            true,
        );
    }
    let chain_ref = access_chain_to_ref(projected_ref, element_access);
    (chain_ref.expr, chain_ref.has_option)
}

fn projected_leaf_expr(
    raw_bind: &syn::Ident,
    projection: &LayerProjection<'_>,
    element_access: &AccessChain,
) -> (TokenStream, bool) {
    let raw_ref = quote! { #raw_bind };
    if projection.parent_access.option_layers() > 0 {
        let tuple_ref = access_chain_to_option_ref(&raw_ref, projection.parent_access);
        let param = idents::tuple_proj_param();
        let projected_ref = project_tuple_element_ref(&quote! { #param }, projection);
        if element_access.option_layers() > 0 {
            let elem_ref = access_chain_to_option_ref(&projected_ref, element_access);
            (quote! { (#tuple_ref).and_then(|#param| #elem_ref) }, true)
        } else {
            let elem_ref = access_chain_to_ref(&projected_ref, element_access).expr;
            (quote! { (#tuple_ref).map(|#param| #elem_ref) }, true)
        }
    } else {
        let tuple_ref = access_chain_to_ref(&raw_ref, projection.parent_access).expr;
        let projected_ref = project_tuple_element_ref(&tuple_ref, projection);
        apply_element_access(&projected_ref, element_access)
    }
}

fn projected_leaf_body(
    vec_bind: &TokenStream,
    projection: &LayerProjection<'_>,
    element_access: &AccessChain,
    per_elem_push: &TokenStream,
) -> TokenStream {
    let raw_bind = idents::leaf_value_raw();
    let leaf_bind = idents::leaf_value();
    let (leaf_expr, has_option) = projected_leaf_expr(&raw_bind, projection, element_access);
    if has_option {
        quote! {
            for #raw_bind in #vec_bind.iter() {
                let #leaf_bind: ::std::option::Option<_> = #leaf_expr;
                #per_elem_push
            }
        }
    } else {
        quote! {
            for #raw_bind in #vec_bind.iter() {
                let #leaf_bind = #leaf_expr;
                #per_elem_push
            }
        }
    }
}

fn projected_scan_leaf_body<'a>(
    shape: &'a VecLayers,
    projection: &'a LayerProjection<'a>,
    projection_access: Option<&'a AccessChain>,
    per_elem_push: &'a TokenStream,
) -> impl Fn(&TokenStream) -> TokenStream + 'a {
    let leaf_bind = idents::leaf_value();
    move |vec_bind: &TokenStream| -> TokenStream {
        if let Some(element_access) = projection_access {
            return projected_leaf_body(vec_bind, projection, element_access, per_elem_push);
        }
        if shape.inner_access.is_empty() || shape.inner_access.is_single_plain_option() {
            quote! {
                for #leaf_bind in #vec_bind.iter() {
                    #per_elem_push
                }
            }
        } else {
            let raw_bind = idents::leaf_value_raw();
            let chain_ref = access_chain_to_ref(&quote! { #raw_bind }, &shape.inner_access);
            let resolved = chain_ref.expr;
            if chain_ref.has_option {
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind: ::std::option::Option<_> = #resolved;
                        #per_elem_push
                    }
                }
            } else {
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind = #resolved;
                        #per_elem_push
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_projected_vec_primitive(
    parent_access: &TokenStream,
    shape: &VecLayers,
    projection: ParentVecProjection<'_>,
    leaf: crate::ir::PrimitiveLeaf<'_>,
    idx: usize,
    column_name: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let series_local = idents::vec_field_series(idx);
    let named = idents::field_named_series();
    let columns = idents::columns();
    let leaf_arr = idents::leaf_arr();
    let total_leaves = idents::total_leaves();

    let layers = projected_layer_idents(idx, shape.depth());
    let layer_counters = projected_layer_counters(idx, shape.depth());
    let dummy_access = TokenStream::new();
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &dummy_access,
            idx,
            name: column_name,
        },
        decimal128_encode_trait: &config.traits.decimal128_encode,
        paths: &config.external_paths,
    };
    let pep = super::vec::pep_for_primitive_leaf(leaf, &leaf_ctx, shape);
    let leaf_projection_access = leaf_projection_access(shape, &projection);

    let emitter = ShapeEmitter::tuple(
        shape,
        parent_access,
        &layers,
        &total_leaves,
        &layer_counters,
        pp,
        pa_root,
        Some(projection.projection),
    );
    let precount = emitter.precount();
    let leaf_body = projected_scan_leaf_body(
        shape,
        &projection.projection,
        leaf_projection_access.as_ref(),
        &pep.per_elem_push,
    );
    let scan = emitter.scan(&leaf_body, &pep.leaf_offsets_post_push);
    let offsets_decls = emitter.offsets_decls();
    let validity_decls = emitter.validity_decls();
    let materialize = projected_materialize(
        &emitter,
        &leaf_arr,
        &pep.leaf_logical_dtype,
        &pep.leaf_arr_expr,
        pp,
    );

    let extra_imports = pep.extra_imports;
    let storage_decls = pep.storage_decls;

    quote! {
        {
            let #series_local: #pp::Series = {
                #extra_imports
                #precount
                #storage_decls
                #offsets_decls
                #validity_decls
                #scan
                #materialize
            };
            let #named = #series_local.with_name(#column_name.into());
            #columns.push(#named.into());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_projected_vec_nested(
    parent_access: &TokenStream,
    shape: &VecLayers,
    projection: ParentVecProjection<'_>,
    type_path: &TokenStream,
    idx: usize,
    column_name: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let total_leaves = idents::nested_total(idx);
    let flat = idents::nested_flat(idx);
    let positions = idents::nested_positions(idx);

    let layers = projected_layer_idents(idx, shape.depth());
    let layer_counters = projected_layer_counters(idx, shape.depth());
    let emitter = ShapeEmitter::tuple(
        shape,
        parent_access,
        &layers,
        &total_leaves,
        &layer_counters,
        pp,
        pa_root,
        Some(projection.projection),
    );
    let precount = emitter.precount();
    let has_inner_option = shape.has_inner_option();

    let leaf_v = idents::leaf_value();
    let inner_v = idents::tuple_nested_inner_v();
    let per_elem_push = if has_inner_option {
        let flat_idx = idx_size_len_expr(&flat, pp);
        quote! {
            match #leaf_v {
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
            #flat.push(#leaf_v);
        }
    };
    let leaf_offsets_post_push = if has_inner_option {
        quote! { #positions.len() }
    } else {
        quote! { #flat.len() }
    };
    let leaf_projection_access = leaf_projection_access(shape, &projection);
    let leaf_body = projected_scan_leaf_body(
        shape,
        &projection.projection,
        leaf_projection_access.as_ref(),
        &per_elem_push,
    );
    let scan = emitter.scan(&leaf_body, &leaf_offsets_post_push);
    let offsets_decls = emitter.offsets_decls();
    let validity_decls = emitter.validity_decls();

    let positions_decl = if has_inner_option {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total_leaves);
        }
    } else {
        TokenStream::new()
    };

    let dispatch = materialize_nested_columns(&NestedMaterializeCtx {
        field_idx: idx,
        ty: type_path,
        column_prefix: column_name,
        flat: &flat,
        positions: has_inner_option.then_some(&positions),
        total_len: quote! { #total_leaves },
        wrapper: NestedWrapper::List {
            shape,
            layers: &layers,
            arr_id_for_layer: idents::tuple_layer_list_arr,
        },
        columnar_trait: &config.traits.columnar,
        to_df_trait: &config.traits.to_dataframe,
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

fn projected_layer_idents(idx: usize, depth: usize) -> Vec<LayerIdents> {
    (0..depth)
        .map(|layer| LayerIdents::new(idents::LayerNamespace::Tuple { field_idx: idx }, layer))
        .collect()
}

fn projected_layer_counters(idx: usize, depth: usize) -> Vec<syn::Ident> {
    (0..depth.saturating_sub(1))
        .map(|layer| idents::tuple_layer_total(idx, layer))
        .collect()
}

fn projected_materialize(
    emitter: &ShapeEmitter<'_>,
    leaf_arr: &syn::Ident,
    leaf_logical_dtype: &TokenStream,
    leaf_arr_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let pa_root = emitter.pa_root;
    let seed_arrow_dtype_id = idents::seed_arrow_dtype();
    let seed_dtype_decl = quote! {
        let #seed_arrow_dtype_id: #pa_root::datatypes::ArrowDataType =
            #pa_root::array::Array::dtype(&#leaf_arr).clone();
    };
    let wrap_layers = emitter.layer_wraps_move();
    let stack = shape_assemble_list_stack(
        quote! { ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef },
        quote! { #seed_arrow_dtype_id },
        &wrap_layers,
        leaf_logical_dtype.clone(),
        pp,
        pa_root,
        &idents::tuple_layer_list_arr,
    );
    quote! {
        #leaf_arr_expr
        #seed_dtype_decl
        #stack
    }
}
