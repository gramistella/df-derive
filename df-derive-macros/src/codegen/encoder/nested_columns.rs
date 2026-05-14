use proc_macro2::TokenStream;
use quote::quote;

use super::idents;

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
