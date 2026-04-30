use crate::ir::{BaseType, PrimitiveTransform, Wrapper};
use proc_macro2::TokenStream;
use quote::quote;

pub struct TypeMapping {
    pub rust_element_type: TokenStream,
    pub element_dtype: TokenStream,
    pub full_dtype: TokenStream,
}

pub fn compute_mapping(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TypeMapping {
    let (rust_elem, elem_dtype) = base_and_transform_to_rust_and_dtype(base, transform);
    let full_dtype = wrap_dtype(&elem_dtype, wrappers);
    TypeMapping {
        rust_element_type: rust_elem,
        element_dtype: elem_dtype,
        full_dtype,
    }
}

fn base_and_transform_to_rust_and_dtype(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
) -> (TokenStream, TokenStream) {
    // Attribute stringification (`as_string`) and borrowing (`as_str`) both
    // materialize as a String dtype. The borrowing path emits `Vec<&str>`
    // buffers directly and bypasses this fallback element type, but keeping
    // them aligned means a stray code path that doesn't yet handle `AsStr`
    // degrades to allocating, not panicking.
    if transform
        .is_some_and(|t| matches!(*t, PrimitiveTransform::ToString | PrimitiveTransform::AsStr))
    {
        return (
            quote! { ::std::string::String },
            quote! { polars::prelude::DataType::String },
        );
    }

    match base {
        BaseType::String => (
            quote! { ::std::string::String },
            quote! { polars::prelude::DataType::String },
        ),
        BaseType::F64 => (
            quote! { f64 },
            quote! { polars::prelude::DataType::Float64 },
        ),
        BaseType::F32 => (
            quote! { f32 },
            quote! { polars::prelude::DataType::Float32 },
        ),
        BaseType::I8 => (quote! { i8 }, quote! { polars::prelude::DataType::Int8 }),
        BaseType::U8 => (quote! { u8 }, quote! { polars::prelude::DataType::UInt8 }),
        BaseType::I16 => (quote! { i16 }, quote! { polars::prelude::DataType::Int16 }),
        BaseType::U16 => (quote! { u16 }, quote! { polars::prelude::DataType::UInt16 }),
        BaseType::I64 | BaseType::ISize => {
            (quote! { i64 }, quote! { polars::prelude::DataType::Int64 })
        }
        BaseType::U64 | BaseType::USize => {
            (quote! { u64 }, quote! { polars::prelude::DataType::UInt64 })
        }
        BaseType::U32 => (quote! { u32 }, quote! { polars::prelude::DataType::UInt32 }),
        BaseType::I32 => (quote! { i32 }, quote! { polars::prelude::DataType::Int32 }),
        BaseType::Bool => (
            quote! { bool },
            quote! { polars::prelude::DataType::Boolean },
        ),
        BaseType::DateTimeUtc => (
            // we materialize as i64 then cast to Datetime dtype later
            quote! { i64 },
            quote! { polars::prelude::DataType::Datetime(polars::prelude::TimeUnit::Milliseconds, None) },
        ),
        BaseType::Decimal => (
            // we materialize as String then cast to Decimal dtype later
            quote! { ::std::string::String },
            quote! { polars::prelude::DataType::Decimal(38, 10) },
        ),
        BaseType::Struct(..) | BaseType::Generic(_) => {
            (quote! { () }, quote! { polars::prelude::DataType::Null })
        }
    }
}

fn wrap_dtype(element_dtype: &TokenStream, wrappers: &[Wrapper]) -> TokenStream {
    if wrappers.iter().any(|w| matches!(w, Wrapper::Vec)) {
        quote! { polars::prelude::DataType::List(Box::new(#element_dtype)) }
    } else {
        quote! { #element_dtype }
    }
}

/// Dtype of one element of the *outermost* list in `wrappers`, used to
/// construct the per-field `ListBuilder` for nested-Vec shapes. The inner
/// Series fed to that builder has this dtype:
///
/// - `Vec<T>` → element dtype (the list contains `T` directly).
/// - `Vec<Vec<T>>` → `List<element>` (each element is itself a list of `T`).
/// - `Vec<Vec<Vec<T>>>` → `List<List<element>>`, etc.
/// - `Option<Vec<T>>` / `Vec<Option<T>>` → element dtype (`Option` doesn't
///   add a list layer; nullability is carried by the values, not a wrapper).
///
/// The macro's `schema()` reporting wraps `element` exactly once for any
/// `Vec`-containing field (a known limitation), so the outer Series's
/// runtime dtype can be deeper than the reported schema dtype. The list
/// builder needs the runtime dtype of its appended Series, not the schema's
/// flattened version, or strict-typed builders like
/// `ListPrimitiveChunkedBuilder` reject the append with a `SchemaMismatch`.
pub fn outer_list_inner_dtype(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TokenStream {
    let (_, element_dtype) = base_and_transform_to_rust_and_dtype(base, transform);
    let vec_count = wrappers
        .iter()
        .filter(|w| matches!(w, Wrapper::Vec))
        .count();
    let mut dt = element_dtype;
    for _ in 0..vec_count.saturating_sub(1) {
        dt = quote! { polars::prelude::DataType::List(Box::new(#dt)) };
    }
    dt
}

pub fn needs_cast(transform: Option<&PrimitiveTransform>) -> bool {
    transform.is_some_and(|t| match t {
        PrimitiveTransform::DateTimeToMillis | PrimitiveTransform::DecimalToString => true,
        PrimitiveTransform::ToString | PrimitiveTransform::AsStr => false,
    })
}

pub fn map_primitive_expr(
    var: &TokenStream,
    transform: Option<&PrimitiveTransform>,
) -> TokenStream {
    transform.map_or_else(
        || quote! { (#var).clone() },
        |t| match *t {
            PrimitiveTransform::DateTimeToMillis => quote! { (#var).timestamp_millis() },
            PrimitiveTransform::ToString | PrimitiveTransform::DecimalToString => {
                quote! { (#var).to_string() }
            }
            PrimitiveTransform::AsStr => {
                // Allocating fallback for codegen sites that can't use a
                // `Vec<&str>` columnar buffer — in practice this is only
                // hit at the leaf of stacked-Option shapes (e.g.
                // `Option<Option<T>>`), where the buffer type
                // `Vec<Option<String>>` flattens both layers anyway. All
                // `Vec<…>` shapes route around this arm via
                // `generate_inner_series_from_vec`, which builds a
                // `Vec<&str>` directly. Emitting valid Rust here keeps the
                // per-field `AsRef<str>` const-fn assert as the canonical
                // user-visible error rather than a proc-macro internal
                // panic.
                quote! { <_ as ::core::convert::AsRef<str>>::as_ref(&(#var)).to_string() }
            }
        },
    )
}
