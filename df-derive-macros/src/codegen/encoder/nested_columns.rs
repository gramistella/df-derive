use proc_macro2::TokenStream;
use quote::quote;

use crate::codegen::external_paths::ExternalPaths;
use crate::ir::VecLayers;

use super::idents;
use super::shape_walk::{
    LayerIdents, shape_assemble_list_stack, shape_freeze_offsets_buffers,
    shape_freeze_validity_bitmaps, shape_layer_wraps_clone,
};

pub(super) struct NestedMaterializeCtx<'a> {
    pub field_idx: usize,
    pub ty: &'a TokenStream,
    pub column_prefix: &'a str,
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

#[derive(Clone, Copy)]
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
    ty: &TokenStream,
    columnar_trait: &syn::Path,
    flat: &syn::Ident,
) -> TokenStream {
    let validate_nested_frame = idents::validate_nested_frame();
    quote! {
        let #df = <#ty as #columnar_trait>::columnar_from_refs(&#flat)?;
        #validate_nested_frame(&#df, #flat.len(), ::core::any::type_name::<#ty>())?;
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
    pub col_name: &'a syn::Ident,
    pub dtype: &'a syn::Ident,
    pub inner_full: &'a syn::Ident,
}

pub(super) fn inner_col_direct(ids: NestedColumnIdents<'_>) -> TokenStream {
    let validate_nested_column_dtype = idents::validate_nested_column_dtype();
    let NestedColumnIdents {
        df,
        col_name,
        dtype,
        inner_full,
        ..
    } = ids;
    quote! {{
        let #inner_full = #df.column(#col_name)?.as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
        #inner_full.clone()
    }}
}

pub(super) fn inner_col_take(ids: NestedColumnIdents<'_>) -> TokenStream {
    let validate_nested_column_dtype = idents::validate_nested_column_dtype();
    let NestedColumnIdents {
        df,
        take,
        col_name,
        dtype,
        inner_full,
    } = ids;
    quote! {{
        let #inner_full = #df
            .column(#col_name)?
            .as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
        #inner_full.take(&#take)?
    }}
}

pub(super) fn inner_col_empty(dtype: &syn::Ident, pp: &TokenStream) -> TokenStream {
    quote! {
        #pp::Series::new_empty("".into(), #dtype)
    }
}

pub(super) fn inner_col_all_absent(
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
    let take = idents::nested_take(ctx.field_idx);
    let columns = idents::columns();
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let inner_full = idents::nested_inner_full();

    let column_idents = NestedColumnIdents {
        df: &df,
        take: &take,
        col_name: &col_name,
        dtype: &dtype,
        inner_full: &inner_full,
    };
    let inner_col_direct = inner_col_direct(column_idents);
    let inner_col_take = inner_col_take(column_idents);
    let inner_col_empty = inner_col_empty(&dtype, pp);
    let inner_col_all_absent = inner_col_all_absent(&dtype, &ctx.total_len, pp);

    let series_direct = wrap_nested_column(&ctx.wrapper, &inner_col_direct, &dtype, pp, pa_root);
    let series_take = wrap_nested_column(&ctx.wrapper, &inner_col_take, &dtype, pp, pa_root);
    let series_empty = wrap_nested_column(&ctx.wrapper, &inner_col_empty, &dtype, pp, pa_root);
    let series_all_absent =
        wrap_nested_column(&ctx.wrapper, &inner_col_all_absent, &dtype, pp, pa_root);

    let consume_direct = consume_nested_columns(
        &columns,
        ctx.column_prefix,
        ctx.to_df_trait,
        ctx.ty,
        &series_direct,
        pp,
    );
    let consume_take = consume_nested_columns(
        &columns,
        ctx.column_prefix,
        ctx.to_df_trait,
        ctx.ty,
        &series_take,
        pp,
    );
    let consume_empty = consume_nested_columns(
        &columns,
        ctx.column_prefix,
        ctx.to_df_trait,
        ctx.ty,
        &series_empty,
        pp,
    );
    let consume_all_absent = consume_nested_columns(
        &columns,
        ctx.column_prefix,
        ctx.to_df_trait,
        ctx.ty,
        &series_all_absent,
        pp,
    );

    let df_decl = nested_df_decl(&df, ctx.ty, ctx.columnar_trait, ctx.flat);
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
            df_decl,
            take_decl,
            consume_direct,
            consume_take,
            consume_empty,
            consume_all_absent,
        },
    )
}

/// Build the per-column emit body that iterates `<T as ToDataFrame>::schema()?`
/// and pushes each inner-Series-yielding expression onto `columns` with the
/// parent name prefixed. Shared by every nested dispatch arm, with each arm
/// supplying a different per-column inner-Series expression.
pub(super) fn consume_nested_columns(
    columns: &syn::Ident,
    parent_name: &str,
    to_df_trait: &syn::Path,
    ty: &TokenStream,
    series_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let prefixed = idents::nested_prefixed_name();
    let inner = idents::nested_inner_series();
    let named = idents::field_named_series();
    quote! {
        for (#col_name, #dtype) in
            <#ty as #to_df_trait>::schema()?
        {
            let #col_name: &str = #col_name.as_str();
            let #dtype: &#pp::DataType = &#dtype;
            {
                let #prefixed = ::std::format!(
                    "{}.{}", #parent_name, #col_name,
                );
                let #inner: #pp::Series = #series_expr;
                let #named = #inner
                    .with_name(#prefixed.as_str().into());
                #columns.push(#named.into());
            }
        }
    }
}
