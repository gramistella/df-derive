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
    // Attribute stringification always materializes as String dtype
    if transform.is_some_and(|t| matches!(*t, PrimitiveTransform::ToString)) {
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
        BaseType::Struct(_ident) => (quote! { () }, quote! { polars::prelude::DataType::Null }),
    }
}

fn wrap_dtype(element_dtype: &TokenStream, wrappers: &[Wrapper]) -> TokenStream {
    if wrappers.iter().any(|w| matches!(w, Wrapper::Vec)) {
        quote! { polars::prelude::DataType::List(Box::new(#element_dtype)) }
    } else {
        quote! { #element_dtype }
    }
}

pub fn needs_cast(transform: Option<&PrimitiveTransform>) -> bool {
    transform.is_some_and(|t| {
        matches!(
            *t,
            PrimitiveTransform::DateTimeToMillis | PrimitiveTransform::DecimalToString
        )
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
        },
    )
}
