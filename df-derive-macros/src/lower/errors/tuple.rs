use proc_macro2::Span;

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
