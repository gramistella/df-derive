use quote::ToTokens;

pub fn unannotated_cow_bytes<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
    can_add_as_binary: bool,
) -> syn::Error {
    let message = if can_add_as_binary {
        format!(
            "field `{field_display_name}` uses `Cow<'_, [u8]>` without `as_binary`; \
             add `#[df_derive(as_binary)]` to encode it as Binary, or use \
             `Vec<u8>` if you want the default `List(UInt8)` representation"
        )
    } else {
        format!(
            "field `{field_display_name}` contains `Cow<'_, [u8]>` in a tuple element; \
             tuple elements cannot be annotated with `as_binary`, so use `Vec<u8>` for \
             the default `List(UInt8)` representation or hoist the bytes into a named \
             struct field with `#[df_derive(as_binary)]`"
        )
    };
    syn::Error::new_spanned(span, message)
}

pub fn unannotated_borrowed_bytes<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
    can_add_as_binary: bool,
) -> syn::Error {
    let message = if can_add_as_binary {
        format!(
            "field `{field_display_name}` uses `&[u8]` without `as_binary`; \
             add `#[df_derive(as_binary)]` to encode it as Binary, or use \
             `Vec<u8>` if you want the default `List(UInt8)` representation"
        )
    } else {
        format!(
            "field `{field_display_name}` contains `&[u8]` in a tuple element; \
             tuple elements cannot be annotated with `as_binary`, so use `Vec<u8>` \
             for the default `List(UInt8)` representation or hoist the bytes into \
             a named struct field with `#[df_derive(as_binary)]`"
        )
    };
    syn::Error::new_spanned(span, message)
}

pub fn borrowed_slice<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` uses `&[T]`, but df-derive only \
             supports `&[u8]` with `#[df_derive(as_binary)]`; use `Vec<T>` \
             for list columns"
        ),
    )
}

pub fn cow_slice<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` uses `Cow<'_, [T]>`, but df-derive only \
             supports `Cow<'_, [u8]>` with `#[df_derive(as_binary)]`; use `Vec<T>` \
             for list columns"
        ),
    )
}

pub fn as_str_wrong_base<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_str` but its base type does not implement \
             `AsRef<str>`; `as_str` only applies to `String`, `&str`, `Cow<'_, str>`, \
             custom struct types, or generic type parameters — drop the attribute or \
             change the field type"
        ),
    )
}

pub fn as_string_std_duration<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_string`, but \
             `std::time::Duration` and `core::time::Duration` do not implement \
             `Display`; drop `as_string` to encode a Duration column, or wrap \
             the value in a custom type that implements `Display`"
        ),
    )
}

pub fn as_string_bytes<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_string`, but byte slices \
             (`&[u8]`/`Cow<'_, [u8]>`) do not implement `Display`; use \
             `#[df_derive(as_binary)]` for a Binary column, use `Vec<u8>` \
             for a `List(UInt8)` column, or wrap the value in a custom type \
             that implements `Display`"
        ),
    )
}

pub fn as_string_slice<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_string`, but borrowed slices \
             (`&[T]`/`Cow<'_, [T]>`) do not implement `Display`; use `Vec<T>` \
             for list columns, or wrap the value in a custom type that \
             implements `Display`"
        ),
    )
}

pub fn as_string_tuple<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `as_string` but its type is a tuple; \
             field-level attributes do not apply to multi-column tuple fields. \
             Hoist the tuple into a named struct that derives `ToDataFrame` if \
             you need per-element attributes."
        ),
    )
}

pub fn decimal_wrong_base<S: ToTokens + ?Sized>(span: &S, field_display_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `decimal(...)` but its base type is not \
             a decimal backend candidate; `decimal(...)` applies to types named \
             `Decimal`, custom struct types, or generic type parameters that \
             implement `Decimal128Encode`"
        ),
    )
}

pub fn time_unit_naive_date<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
             `chrono::NaiveDate`, which has a fixed encoding (i32 days since 1970-01-01) \
             and offers no unit choice — remove the attribute"
        ),
    )
}

pub fn time_unit_naive_time<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
             `chrono::NaiveTime`, which has a fixed encoding (i64 nanoseconds since \
             midnight) and offers no unit choice — remove the attribute"
        ),
    )
}

pub fn time_unit_wrong_base<S: ToTokens + ?Sized>(
    span: &S,
    field_display_name: &str,
) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "field `{field_display_name}` has `time_unit = \"...\"` but its base type is \
             not `chrono::DateTime<Tz>`, `chrono::NaiveDateTime`, \
             `std::time::Duration`, `core::time::Duration`, or \
             `chrono::Duration`; remove the attribute or change the field type"
        ),
    )
}
