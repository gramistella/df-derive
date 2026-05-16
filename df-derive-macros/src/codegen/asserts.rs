use crate::codegen::encoder::idents;
use crate::ir::{DecimalBackend, DisplayBase, LeafSpec, StringyBase, StructIR};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Type;

use super::type_deps::{GenericContext, push_unique_type, type_depends_on_generics};

pub fn generate_eager_asserts(
    ir: &StructIR,
    to_dataframe_trait: &syn::Path,
    columnar_trait: &syn::Path,
    decimal128_encode_trait: &syn::Path,
) -> TokenStream {
    let generic_ctx = GenericContext::new(ir);
    let mut nested_types = Vec::new();
    let mut decimal_backend_types = Vec::new();
    let mut as_ref_str_types = Vec::new();
    let mut display_types = Vec::new();

    for field in &ir.fields {
        collect_nested_asserts(&field.leaf_spec, &generic_ctx, &mut nested_types);
        collect_decimal_asserts(&field.leaf_spec, &generic_ctx, &mut decimal_backend_types);
        collect_as_ref_str_asserts(&field.leaf_spec, &generic_ctx, &mut as_ref_str_types);
        collect_display_asserts(&field.leaf_spec, &generic_ctx, &mut display_types);
    }

    if nested_types.is_empty()
        && decimal_backend_types.is_empty()
        && as_ref_str_types.is_empty()
        && display_types.is_empty()
    {
        return TokenStream::new();
    }

    let nested_asserts = if nested_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_nested_traits = idents::nested_traits_assert_helper();
        quote! {
            const fn #assert_nested_traits<
                __DfDeriveT: #to_dataframe_trait + #columnar_trait
            >() {}

            #(
                #assert_nested_traits::<#nested_types>();
            )*
        }
    };
    let decimal_backend_asserts = if decimal_backend_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_decimal_backend = idents::decimal_backend_assert_helper();
        quote! {
            const fn #assert_decimal_backend<
                __DfDeriveT: #decimal128_encode_trait
            >() {}

            #(
                #assert_decimal_backend::<#decimal_backend_types>();
            )*
        }
    };
    let as_ref_str_asserts = if as_ref_str_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_as_ref_str = idents::as_ref_str_assert_helper();
        quote! {
            const fn #assert_as_ref_str<
                __DfDeriveT: ?::core::marker::Sized + ::core::convert::AsRef<str>
            >() {}

            #(
                #assert_as_ref_str::<#as_ref_str_types>();
            )*
        }
    };
    let display_asserts = if display_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_display = idents::display_assert_helper();
        quote! {
            const fn #assert_display<
                __DfDeriveT: ?::core::marker::Sized + ::core::fmt::Display
            >() {}

            #(
                #assert_display::<#display_types>();
            )*
        }
    };

    quote! {
        #nested_asserts
        #decimal_backend_asserts
        #as_ref_str_asserts
        #display_asserts
    }
}

fn collect_decimal_asserts(leaf: &LeafSpec, generic_ctx: &GenericContext, out: &mut Vec<Type>) {
    leaf.walk_terminal_leaves(&mut |leaf| {
        if let LeafSpec::Decimal {
            backend: DecimalBackend::Struct(ty),
            ..
        } = leaf
            && !type_depends_on_generics(ty, generic_ctx)
        {
            push_unique_type(out, ty);
        }
    });
}

fn collect_nested_asserts(leaf: &LeafSpec, generic_ctx: &GenericContext, out: &mut Vec<Type>) {
    leaf.walk_terminal_leaves(&mut |leaf| {
        if let LeafSpec::Struct(ty) = leaf
            && !type_depends_on_generics(ty, generic_ctx)
        {
            push_unique_type(out, ty);
        }
    });
}

fn collect_as_ref_str_asserts(leaf: &LeafSpec, generic_ctx: &GenericContext, out: &mut Vec<Type>) {
    leaf.walk_terminal_leaves(&mut |leaf| {
        if let LeafSpec::AsStr(StringyBase::Struct(ty)) = leaf
            && !type_depends_on_generics(ty, generic_ctx)
        {
            push_unique_type(out, ty);
        }
    });
}

fn collect_display_asserts(leaf: &LeafSpec, generic_ctx: &GenericContext, out: &mut Vec<Type>) {
    leaf.walk_terminal_leaves(&mut |leaf| {
        if let LeafSpec::AsString(DisplayBase::Struct(ty)) = leaf
            && !type_depends_on_generics(ty, generic_ctx)
        {
            push_unique_type(out, ty);
        }
    });
}
