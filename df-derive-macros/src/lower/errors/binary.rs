use quote::ToTokens;

pub fn bare_binary_u8<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_binary` but its type is a single `u8`; \
             `as_binary` requires a `Vec<u8>` shape — bare `u8` is a single byte, not \
             a binary blob. Wrap the field in `Vec<u8>` to opt into Binary."
        ),
    )
}

pub fn binary_inner_option<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_binary` but the wrapper stack places an \
             `Option` between the `Vec` and the `u8` leaf; BinaryView cannot carry \
             per-byte nulls. Use `Vec<u8>` directly (drop the inner `Option`)."
        ),
    )
}

pub fn binary_wrong_base<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_binary` but its base type is not `u8`; \
             `as_binary` requires a `Vec<u8>` shape (the innermost `Vec` becomes the \
             Binary blob). Change the field type or drop the attribute."
        ),
    )
}

pub fn binary_cow_slice<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_binary` on `Cow<'_, [T]>`, but \
             `as_binary` only supports `Cow<'_, [u8]>`; use `Vec<T>` for list columns"
        ),
    )
}

pub fn binary_borrowed_slice<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_binary` on `&[T]`, but \
             `as_binary` only supports `&[u8]`; use `Vec<T>` for list columns"
        ),
    )
}
