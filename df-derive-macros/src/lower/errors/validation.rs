use quote::ToTokens;

pub fn direct_self_reference<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
    struct_name: &syn::Ident,
) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` recursively references `{struct_name}` after \
             transparent wrapper peeling; recursive nested DataFrame schemas are not \
             supported. Store an identifier or foreign key, flatten the structure \
             before deriving, or mark the field `#[df_derive(skip)]`."
        ),
    )
}
