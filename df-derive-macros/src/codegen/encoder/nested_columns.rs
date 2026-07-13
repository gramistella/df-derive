use proc_macro2::TokenStream;
use quote::quote;

use crate::codegen::external_paths::ExternalPaths;
use crate::ir::{NestedNamePolicy, VecLayers};

use super::idents::{self, LayerIdents};
use super::shape_walk::{
    shape_assemble_list_stack, shape_freeze_offsets_buffers, shape_freeze_validity_bitmaps,
    shape_layer_wraps_clone,
};

pub(super) struct NestedMaterializeCtx<'a> {
    pub field_idx: usize,
    pub ty: &'a TokenStream,
    pub column_prefix: &'a str,
    pub name_policy: &'a NestedNamePolicy,
    pub flat: &'a syn::Ident,
    pub positions: Option<&'a syn::Ident>,
    pub total_len: TokenStream,
    pub wrapper: NestedWrapper<'a>,
    pub columnar_trait: &'a syn::Path,
    pub to_df_trait: &'a syn::Path,
    pub paths: &'a ExternalPaths,
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
    pub df_decl: TokenStream,
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
        df_decl,
        take_decl,
        consume_direct,
        consume_take,
        consume_empty,
        consume_all_absent,
    } = branches;

    match kind {
        NestedMaterializeKind::LeafBare => {
            quote! {
                #df_decl
                #consume_direct
            }
        }
        NestedMaterializeKind::LeafOptional => {
            quote! {
                if #flat.is_empty() {
                    #consume_all_absent
                } else if #flat.len() == #total_len {
                    #df_decl
                    #consume_direct
                } else {
                    #df_decl
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
                    #df_decl
                    #offsets_freeze
                    #consume_direct
                }
            }
        }
    }
}

pub(super) fn nested_df_decl(
    df: &syn::Ident,
    schema: &syn::Ident,
    ty: &TokenStream,
    columnar_trait: &syn::Path,
    flat: &syn::Ident,
) -> TokenStream {
    let validate_nested_frame = idents::validate_nested_frame();
    quote! {
        let #df = <#ty as #columnar_trait>::encode(#flat.as_slice())?;
        #validate_nested_frame(
            &#df,
            &#schema,
            #flat.len(),
            ::core::any::type_name::<#ty>(),
        )?;
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

#[derive(Clone, Copy)]
pub(super) struct NestedColumnIdents<'a> {
    pub df: &'a syn::Ident,
    pub take: &'a syn::Ident,
    pub index: &'a syn::Ident,
    pub inner_full: &'a syn::Ident,
}

pub(super) fn build_inner_col_direct(ids: NestedColumnIdents<'_>) -> TokenStream {
    let NestedColumnIdents {
        df,
        index,
        inner_full,
        ..
    } = ids;
    quote! {{
        let #inner_full = #df.columns()[#index].as_materialized_series();
        #inner_full.clone()
    }}
}

pub(super) fn build_inner_col_take(ids: NestedColumnIdents<'_>) -> TokenStream {
    let NestedColumnIdents {
        df,
        take,
        index,
        inner_full,
    } = ids;
    quote! {{
        let #inner_full = #df
            .columns()[#index]
            .as_materialized_series();
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

fn wrap_nested_column(
    wrapper: &NestedWrapper<'_>,
    inner_col_expr: &TokenStream,
    dtype: &syn::Ident,
    pp: &TokenStream,
    pa_root: &TokenStream,
) -> TokenStream {
    let NestedWrapper::List {
        shape,
        layers,
        arr_id_for_layer,
    } = wrapper
    else {
        return inner_col_expr.clone();
    };
    let inner_chunk = idents::nested_inner_chunk();
    let inner_col = idents::nested_inner_col();
    let inner_rech = idents::nested_inner_rech();
    let chunk_decl = quote! {
        let #inner_col: #pp::Series = #inner_col_expr;
        let #inner_rech = #inner_col.rechunk();
        let #inner_chunk: #pp::ArrayRef = #inner_rech.chunks()[0].clone();
    };
    let wrap_layers = shape_layer_wraps_clone(shape, layers);
    let stack = shape_assemble_list_stack(
        quote! { #inner_chunk },
        quote! { #inner_chunk.dtype().clone() },
        &wrap_layers,
        quote! { (*#dtype).clone() },
        pp,
        pa_root,
        arr_id_for_layer,
    );
    quote! {{
        #chunk_decl
        #stack
    }}
}

pub(super) fn materialize_nested_columns(ctx: &NestedMaterializeCtx<'_>) -> TokenStream {
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();
    let df = idents::nested_df(ctx.field_idx);
    let schema = idents::nested_schema(ctx.field_idx);
    let take = idents::nested_take(ctx.field_idx);
    let columns = idents::columns();
    let index = idents::nested_col_index();
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();

    let column_idents = NestedColumnIdents {
        df: &df,
        take: &take,
        index: &index,
        inner_full: &inner_full,
    };
    let inner_col_direct = build_inner_col_direct(column_idents);
    let inner_col_take = build_inner_col_take(column_idents);
    let inner_col_empty = build_inner_col_empty(&dtype, pp);
    let inner_col_all_absent = build_inner_col_all_absent(&dtype, &ctx.total_len, pp);

    let series_direct = wrap_nested_column(&ctx.wrapper, &inner_col_direct, &dtype, pp, pa_root);
    let series_take = wrap_nested_column(&ctx.wrapper, &inner_col_take, &dtype, pp, pa_root);
    let series_empty = wrap_nested_column(&ctx.wrapper, &inner_col_empty, &dtype, pp, pa_root);
    let series_all_absent =
        wrap_nested_column(&ctx.wrapper, &inner_col_all_absent, &dtype, pp, pa_root);

    let consume = |series| {
        consume_nested_columns(
            &columns,
            ctx.column_prefix,
            ctx.name_policy,
            &schema,
            &index,
            series,
            pp,
        )
    };
    let consume_direct = consume(&series_direct);
    let consume_take = consume(&series_take);
    let consume_empty = consume(&series_empty);
    let consume_all_absent = consume(&series_all_absent);

    let df_decl = nested_df_decl(&df, &schema, ctx.ty, ctx.columnar_trait, ctx.flat);
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

    let dispatch = nested_materialize_dispatch(
        kind,
        ctx.flat,
        &ctx.total_len,
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

    let to_df_trait = ctx.to_df_trait;
    let ty = ctx.ty;
    quote! {
        let #schema = <#ty as #to_df_trait>::schema()?;
        #dispatch
    }
}

pub(super) fn consume_nested_columns(
    columns: &syn::Ident,
    parent_name: &str,
    name_policy: &NestedNamePolicy,
    schema: &syn::Ident,
    index: &syn::Ident,
    series_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let prefixed = idents::nested_prefixed_name();
    let inner = idents::nested_inner_series();
    let named = idents::field_named_series();
    let output_name = crate::codegen::nested_names::compose_nested_name(
        name_policy,
        parent_name,
        &quote! { #col_name },
    );
    quote! {
        for (#index, (#col_name, #dtype)) in #schema.iter().enumerate() {
            let #col_name: &str = #col_name.as_str();
            let #dtype: &#pp::DataType = #dtype;
            {
                let #prefixed = #output_name;
                let #inner: #pp::Series = #series_expr;
                let #named = #inner
                    .with_name(#prefixed.as_str().into());
                #columns.push(#named.into());
            }
        }
    }
}
