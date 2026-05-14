use proc_macro2::TokenStream;
use quote::quote;

use super::idents;

pub(super) fn nested_df_decl(
    df: &syn::Ident,
    ty: &TokenStream,
    columnar_trait: &TokenStream,
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

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(super) struct NestedColumnIdents<'a> {
    pub df: &'a syn::Ident,
    pub take: &'a syn::Ident,
    pub col_name: &'a syn::Ident,
    pub dtype: &'a syn::Ident,
    pub inner_full: &'a syn::Ident,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
pub(super) fn inner_col_empty(dtype: &syn::Ident, pp: &TokenStream) -> TokenStream {
    quote! {
        #pp::Series::new_empty("".into(), #dtype)
    }
}

#[allow(dead_code)]
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

/// Build the per-column emit body that iterates `<T as ToDataFrame>::schema()?`
/// and pushes each inner-Series-yielding expression onto `columns` with the
/// parent name prefixed. Shared by every nested dispatch arm, with each arm
/// supplying a different per-column inner-Series expression.
pub(super) fn consume_nested_columns(
    parent_name: &str,
    to_df_trait: &TokenStream,
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
                columns.push(#named.into());
            }
        }
    }
}
