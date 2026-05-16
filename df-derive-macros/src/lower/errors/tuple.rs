use proc_macro2::Span;
use quote::ToTokens;

fn unsupported_tuple_attr_message(field_display_name: &str, attr: &str) -> String {
    format!(
        "field `{field_display_name}` has `{attr}` but its type is a tuple; \
         field-level attributes do not apply to multi-column tuple fields. \
         Hoist the tuple into a named struct that derives \
         `ToDataFrame` if you need per-element attributes."
    )
}

pub fn unsupported_tuple_attr_at(span: Span, field_display_name: &str, attr: &str) -> syn::Error {
    syn::Error::new(
        span,
        unsupported_tuple_attr_message(field_display_name, attr),
    )
}

pub fn unsupported_wrapped_nested_tuple<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` contains a nested tuple whose projection path \
             is wrapped; nested tuples are supported only when each tuple on that path is \
             unwrapped. Hoist the inner tuple into a named struct deriving `ToDataFrame`, \
             or remove the `Option`/`Vec` wrapper around the tuple."
        ),
    )
}
