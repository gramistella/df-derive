use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ir::{NumericKind, StorageNumericKind};

use crate::codegen::external_paths::ExternalPaths;

pub(in crate::codegen) struct NumericInfo {
    pub native: TokenStream,
    pub dtype: TokenStream,
    pub chunked: TokenStream,
    pub widen_from: Option<TokenStream>,
}

pub(in crate::codegen) fn numeric_info_for(
    kind: NumericKind,
    paths: &ExternalPaths,
) -> NumericInfo {
    let pp = paths.prelude();
    let info = |native: TokenStream, variant: &str, widen_from: Option<TokenStream>| {
        let chunked_ident = format_ident!("{}Chunked", variant);
        let dtype_ident = format_ident!("{}", variant);
        NumericInfo {
            native,
            dtype: quote! { #pp::DataType::#dtype_ident },
            chunked: quote! { #pp::#chunked_ident },
            widen_from,
        }
    };
    match kind.storage_kind() {
        StorageNumericKind::I8 => info(quote! { i8 }, "Int8", None),
        StorageNumericKind::I16 => info(quote! { i16 }, "Int16", None),
        StorageNumericKind::I32 => info(quote! { i32 }, "Int32", None),
        StorageNumericKind::I64 => info(quote! { i64 }, "Int64", None),
        StorageNumericKind::I128 => info(quote! { i128 }, "Int128", None),
        StorageNumericKind::U8 => info(quote! { u8 }, "UInt8", None),
        StorageNumericKind::U16 => info(quote! { u16 }, "UInt16", None),
        StorageNumericKind::U32 => info(quote! { u32 }, "UInt32", None),
        StorageNumericKind::U64 => info(quote! { u64 }, "UInt64", None),
        StorageNumericKind::U128 => info(quote! { u128 }, "UInt128", None),
        StorageNumericKind::F32 => info(quote! { f32 }, "Float32", None),
        StorageNumericKind::F64 => info(quote! { f64 }, "Float64", None),
        StorageNumericKind::ISize => info(quote! { i64 }, "Int64", Some(quote! { isize })),
        StorageNumericKind::USize => info(quote! { u64 }, "UInt64", Some(quote! { usize })),
    }
}

pub(in crate::codegen) fn numeric_stored_value(
    kind: NumericKind,
    source_value: TokenStream,
    native: &TokenStream,
) -> TokenStream {
    let value = if kind.is_nonzero() {
        quote! { (#source_value).get() }
    } else {
        source_value
    };
    if kind.is_widened() {
        quote! { (#value as #native) }
    } else {
        value
    }
}
