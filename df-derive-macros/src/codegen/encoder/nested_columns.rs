use proc_macro2::TokenStream;
use quote::quote;

use crate::codegen::external_paths::ExternalPaths;
use crate::ir::VecLayers;

use super::idents::{self, LayerIdents};
use super::shape_walk::{
    ListAssembly, ListAssemblySeed, ListAssemblyTarget, shape_assemble_list_stack,
    shape_freeze_offsets_buffers, shape_freeze_validity_bitmaps, shape_layer_wraps_clone,
};

pub(super) struct NestedMaterializeCtx<'a> {
    pub field_idx: usize,
    pub ident_scope: idents::GeneratedIdentScope<'a>,
    pub sink: &'a syn::Ident,
    pub ty: &'a TokenStream,
    pub flat: &'a syn::Ident,
    pub positions: Option<&'a syn::Ident>,
    pub total_len: TokenStream,
    pub wrapper: NestedWrapper<'a>,
    pub prefix: Option<SharedListPrefix<'a>>,
    pub columnar_trait: &'a syn::Path,
    pub columnar_spec_trait: &'a syn::Path,
    pub paths: &'a ExternalPaths,
}

#[derive(Clone, Copy)]
pub(super) struct SharedListPrefix<'a> {
    pub shape: &'a VecLayers,
    pub layers: &'a [LayerIdents],
    pub ident_scope: idents::GeneratedIdentScope<'a>,
}

#[derive(Clone, Copy)]
pub(super) enum NestedWrapper<'a> {
    None,
    List {
        shape: &'a VecLayers,
        layers: &'a [LayerIdents],
        arr_id_for_layer: fn(usize) -> syn::Ident,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NestedMaterializeKind {
    LeafBare,
    LeafOptional,
    Vec { has_inner_option: bool },
}

pub(super) struct NestedMaterializeBranches {
    pub validity_freeze: TokenStream,
    pub offsets_freeze: TokenStream,
    pub batch_decl: TokenStream,
    pub take_decl: TokenStream,
    pub consume_direct: TokenStream,
    pub consume_take: TokenStream,
    pub consume_empty: TokenStream,
    pub consume_all_absent: TokenStream,
}

pub(super) fn nested_materialize_dispatch(
    kind: NestedMaterializeKind,
    flat: &syn::Ident,
    total_len: &TokenStream,
    branches: NestedMaterializeBranches,
) -> TokenStream {
    let NestedMaterializeBranches {
        validity_freeze,
        offsets_freeze,
        batch_decl,
        take_decl,
        consume_direct,
        consume_take,
        consume_empty,
        consume_all_absent,
    } = branches;

    match kind {
        NestedMaterializeKind::LeafBare => {
            quote! {
                if #flat.is_empty() {
                    #consume_empty
                } else {
                    #batch_decl
                    #consume_direct
                }
            }
        }
        NestedMaterializeKind::LeafOptional => {
            quote! {
                if #flat.is_empty() {
                    #consume_all_absent
                } else if #flat.len() == #total_len {
                    #batch_decl
                    #consume_direct
                } else {
                    #batch_decl
                    #take_decl
                    #consume_take
                }
            }
        }
        NestedMaterializeKind::Vec {
            has_inner_option: true,
        } => {
            quote! {
                #validity_freeze
                if #total_len == 0 {
                    #offsets_freeze
                    #consume_empty
                } else if #flat.is_empty() {
                    #offsets_freeze
                    #consume_all_absent
                } else if #flat.len() == #total_len {
                    #batch_decl
                    #offsets_freeze
                    #consume_direct
                } else {
                    #batch_decl
                    #take_decl
                    #offsets_freeze
                    #consume_take
                }
            }
        }
        NestedMaterializeKind::Vec {
            has_inner_option: false,
        } => {
            quote! {
                #validity_freeze
                if #flat.is_empty() {
                    #offsets_freeze
                    #consume_empty
                } else {
                    #batch_decl
                    #offsets_freeze
                    #consume_direct
                }
            }
        }
    }
}

pub(super) fn nested_batch_decl(
    columns: &syn::Ident,
    ty: &TokenStream,
    columnar_trait: &syn::Path,
    flat: &syn::Ident,
) -> TokenStream {
    quote! {
        let #columns = <#ty as #columnar_trait>::encode_ref_batch(
            #flat.as_slice(),
        )?.into_columns();
    }
}

pub(super) fn nested_take_decl(
    take: &syn::Ident,
    positions: &syn::Ident,
    pp: &TokenStream,
) -> TokenStream {
    quote! {
        let #take: #pp::IdxCa =
            <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                "".into(),
                #positions.iter().copied(),
            );
    }
}

pub(super) fn build_inner_col_direct(column: &syn::Ident, inner_full: &syn::Ident) -> TokenStream {
    quote! {{
        let #inner_full = #column.as_materialized_series();
        #inner_full.clone()
    }}
}

pub(super) fn build_inner_col_take(
    column: &syn::Ident,
    take: &syn::Ident,
    inner_full: &syn::Ident,
) -> TokenStream {
    quote! {{
        let #inner_full = #column.as_materialized_series();
        #inner_full.take(&#take)?
    }}
}

pub(super) fn build_inner_col_empty(dtype: &syn::Ident, pp: &TokenStream) -> TokenStream {
    quote! {
        #pp::Series::new_empty("".into(), #dtype)
    }
}

pub(super) fn build_inner_col_all_absent(
    dtype: &syn::Ident,
    len: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    quote! {
        #pp::Series::new_empty("".into(), #dtype)
            .extend_constant(#pp::AnyValue::Null, #len)?
    }
}

fn wrap_list_column(
    shape: &VecLayers,
    layers: &[LayerIdents],
    arr_id_for_layer: &dyn Fn(usize) -> syn::Ident,
    inner_col_expr: &TokenStream,
    target: ListAssemblyTarget,
    pp: &TokenStream,
    pa_root: &TokenStream,
) -> TokenStream {
    let inner_chunk = idents::nested_inner_chunk();
    let inner_col = idents::nested_inner_col();
    let inner_rech = idents::nested_inner_rech();
    let logical_dtype = idents::nested_inner_logical_dtype();
    let chunk_decl = quote! {
        let #inner_col: #pp::Series = #inner_col_expr;
        let #logical_dtype: #pp::DataType = #inner_col.dtype().clone();
        let #inner_rech = #inner_col.rechunk();
        let #inner_chunk: #pp::ArrayRef = #inner_rech.chunks()[0].clone();
    };
    let wrap_layers = shape_layer_wraps_clone(shape, layers);
    let stack = shape_assemble_list_stack(ListAssembly {
        seed: ListAssemblySeed {
            payload: quote! { #inner_chunk },
            arrow_dtype: quote! { #inner_chunk.dtype().clone() },
            logical_dtype: quote! { #logical_dtype },
        },
        layers: &wrap_layers,
        target,
        pp,
        pa_root,
        arr_id_for_layer,
    });
    quote! {{
        #chunk_decl
        #stack
    }}
}

fn wrap_nested_column(
    wrapper: &NestedWrapper<'_>,
    prefix: Option<SharedListPrefix<'_>>,
    field_idx: usize,
    inner_col_expr: &TokenStream,
    output_slot: &syn::Ident,
    pp: &TokenStream,
    pa_root: &TokenStream,
) -> TokenStream {
    let terminal_series = match wrapper {
        NestedWrapper::None => inner_col_expr.clone(),
        NestedWrapper::List {
            shape,
            layers,
            arr_id_for_layer,
        } => wrap_list_column(
            shape,
            layers,
            arr_id_for_layer,
            inner_col_expr,
            if prefix.is_none() {
                ListAssemblyTarget::schema_slot(output_slot)
            } else {
                ListAssemblyTarget::intermediate()
            },
            pp,
            pa_root,
        ),
    };

    prefix.map_or_else(
        || match wrapper {
            NestedWrapper::None => {
                quote! { (#terminal_series).with_name(#output_slot.name().clone()) }
            }
            NestedWrapper::List { .. } => terminal_series.clone(),
        },
        |prefix| {
            let arr_id_for_layer =
                |layer| idents::tuple_prefix_list_arr(prefix.ident_scope, field_idx, layer);
            wrap_list_column(
                prefix.shape,
                prefix.layers,
                &arr_id_for_layer,
                &terminal_series,
                ListAssemblyTarget::schema_slot(output_slot),
                pp,
                pa_root,
            )
        },
    )
}

struct NestedSeriesBranches {
    direct: TokenStream,
    take: TokenStream,
    empty: TokenStream,
    all_absent: TokenStream,
}

#[derive(Clone, Copy)]
struct NestedSeriesIdents<'a> {
    output_slot: &'a syn::Ident,
    child_column: &'a syn::Ident,
    take: &'a syn::Ident,
    dtype: &'a syn::Ident,
    inner_full: &'a syn::Ident,
}

fn build_nested_series_branches(
    ctx: &NestedMaterializeCtx<'_>,
    idents: &NestedSeriesIdents<'_>,
    pp: &TokenStream,
    pa_root: &TokenStream,
) -> NestedSeriesBranches {
    let NestedSeriesIdents {
        output_slot,
        child_column,
        take,
        dtype,
        inner_full,
    } = *idents;
    let wrap = |inner| {
        wrap_nested_column(
            &ctx.wrapper,
            ctx.prefix,
            ctx.field_idx,
            &inner,
            output_slot,
            pp,
            pa_root,
        )
    };
    NestedSeriesBranches {
        direct: wrap(build_inner_col_direct(child_column, inner_full)),
        take: wrap(build_inner_col_take(child_column, take, inner_full)),
        empty: wrap(build_inner_col_empty(dtype, pp)),
        all_absent: wrap(build_inner_col_all_absent(dtype, &ctx.total_len, pp)),
    }
}

pub(super) fn materialize_nested_columns(ctx: &NestedMaterializeCtx<'_>) -> TokenStream {
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();
    let child_columns = idents::nested_columns(ctx.field_idx);
    let child_column = idents::nested_column();
    let schema = idents::nested_schema(ctx.field_idx);
    let take = idents::nested_take(ctx.field_idx);
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();
    let output_slot = idents::output_slot(ctx.ident_scope, ctx.field_idx);

    let series = build_nested_series_branches(
        ctx,
        &NestedSeriesIdents {
            output_slot: &output_slot,
            child_column: &child_column,
            take: &take,
            dtype: &dtype,
            inner_full: &inner_full,
        },
        pp,
        pa_root,
    );

    let consume_batch = |series| {
        consume_nested_batch_columns(
            ctx.sink,
            &child_columns,
            &child_column,
            &output_slot,
            series,
            pp,
        )
    };
    let consume_direct = consume_batch(&series.direct);
    let consume_take = consume_batch(&series.take);
    let consume_empty_columns =
        consume_nested_schema_columns(ctx.sink, &schema, &output_slot, &series.empty, pp);
    let consume_all_absent_columns =
        consume_nested_schema_columns(ctx.sink, &schema, &output_slot, &series.all_absent, pp);

    let ty = ctx.ty;
    let columnar_spec_trait = ctx.columnar_spec_trait;
    let schema_decl = quote! {
        let #schema = <#ty as #columnar_spec_trait>::build_schema()?;
    };
    let consume_empty = quote! {
        #schema_decl
        #consume_empty_columns
    };
    let consume_all_absent = quote! {
        #schema_decl
        #consume_all_absent_columns
    };

    let batch_decl = nested_batch_decl(&child_columns, ctx.ty, ctx.columnar_trait, ctx.flat);
    let take_decl = ctx.positions.map_or_else(TokenStream::new, |positions| {
        nested_take_decl(&take, positions, pp)
    });

    let (kind, validity_freeze, offsets_freeze) = match ctx.wrapper {
        NestedWrapper::None => {
            let kind = if ctx.positions.is_some() {
                NestedMaterializeKind::LeafOptional
            } else {
                NestedMaterializeKind::LeafBare
            };
            (kind, TokenStream::new(), TokenStream::new())
        }
        NestedWrapper::List { shape, layers, .. } => (
            NestedMaterializeKind::Vec {
                has_inner_option: ctx.positions.is_some(),
            },
            shape_freeze_validity_bitmaps(shape, layers, pa_root),
            shape_freeze_offsets_buffers(layers, pa_root),
        ),
    };

    nested_materialize_dispatch(
        kind,
        ctx.flat,
        &ctx.total_len,
        NestedMaterializeBranches {
            validity_freeze,
            offsets_freeze,
            batch_decl,
            take_decl,
            consume_direct,
            consume_take,
            consume_empty,
            consume_all_absent,
        },
    )
}

fn nested_column_body(
    sink: &syn::Ident,
    output_slot: &syn::Ident,
    series_expr: &TokenStream,
) -> TokenStream {
    let inner = idents::nested_inner_series();
    quote! {
        let #output_slot = #sink.next_slot()?;
        let #inner = #series_expr;
        #output_slot.commit(#inner.into())?;
    }
}

pub(super) fn consume_nested_batch_columns(
    sink: &syn::Ident,
    columns: &syn::Ident,
    column: &syn::Ident,
    output_slot: &syn::Ident,
    series_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let dtype = idents::nested_col_dtype();
    let body = nested_column_body(sink, output_slot, series_expr);
    quote! {
        for #column in &#columns {
            let #dtype: &#pp::DataType = #column.dtype();
            #body
        }
    }
}

pub(super) fn consume_nested_schema_columns(
    sink: &syn::Ident,
    schema: &syn::Ident,
    output_slot: &syn::Ident,
    series_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let dtype = idents::nested_col_dtype();
    let body = nested_column_body(sink, output_slot, series_expr);
    quote! {
        for (_, #dtype) in #schema.iter() {
            let #dtype: &#pp::DataType = #dtype;
            #body
        }
    }
}
