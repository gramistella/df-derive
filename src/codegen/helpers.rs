use crate::ir::{LeafSpec, StructIR};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Peel transparent wrappers off a field type to find the leaf used in
/// codegen trait-bound asserts. Strips `Option`/`Vec`/`Box`/`Rc`/`Arc`/`Cow`
/// (last-segment ident match) so the assert sees the same leaf type
/// `analyze_type` resolves to. Mirrors the peel in
/// `type_analysis::analyze_type`; if either falls out of sync the
/// `as_str` const-assert points at the wrong type.
fn peel_to_leaf(ty: &syn::Type) -> &syn::Type {
    fn extract_inner<'a>(ty: &'a syn::Type, name: &str) -> Option<&'a syn::Type> {
        if let syn::Type::Path(tp) = ty
            && let Some(seg) = tp.path.segments.last()
            && seg.ident == name
            && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
            && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
        {
            return Some(inner);
        }
        None
    }

    fn extract_cow_inner(ty: &syn::Type) -> Option<&syn::Type> {
        let syn::Type::Path(tp) = ty else {
            return None;
        };
        let seg = tp.path.segments.last()?;
        if seg.ident != "Cow" {
            return None;
        }
        let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
            return None;
        };
        args.args.iter().find_map(|arg| match arg {
            syn::GenericArgument::Type(inner) => Some(inner),
            _ => None,
        })
    }

    let mut cur = ty;
    loop {
        if let Some(inner) = extract_inner(cur, "Option") {
            cur = inner;
            continue;
        }
        if let Some(inner) = extract_inner(cur, "Vec") {
            cur = inner;
            continue;
        }
        if let Some(inner) = extract_inner(cur, "Box") {
            cur = inner;
            continue;
        }
        if let Some(inner) = extract_inner(cur, "Rc") {
            cur = inner;
            continue;
        }
        if let Some(inner) = extract_inner(cur, "Arc") {
            cur = inner;
            continue;
        }
        if let Some(inner) = extract_cow_inner(cur) {
            cur = inner;
            continue;
        }
        break;
    }
    cur
}

pub fn generate_helpers_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let (impl_generics, ty_generics, where_clause) = super::impl_parts_with_bounds(ir, config);

    // Per-field `AsRef<str>` assertions for `as_str` fields. Each assert is
    // a named `const` item inside the impl block (anonymous `const _:` is
    // not allowed in associated-item position). The const-fn form
    // type-checks at monomorphization and gives the user a clean error
    // span on the leaf type (not deep in macro-expanded code). Placement
    // inside the impl is required for generic-leaf fields (`field: T`),
    // where `T` isn't in scope at module level.
    let as_ref_str_asserts: Vec<TokenStream> = ir
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.leaf_spec, LeafSpec::AsStr(_)))
        .map(|(idx, f)| {
            let leaf = peel_to_leaf(&f.field_ty);
            let const_name = format_ident!("_DF_DERIVE_ASSERT_AS_REF_STR_{}", idx);
            quote! {
                #[allow(dead_code)]
                const #const_name: fn() = || {
                    fn _df_derive_assert_as_ref_str<__DfDeriveT: ?::core::marker::Sized + ::core::convert::AsRef<str>>() {}
                    _df_derive_assert_as_ref_str::<#leaf>();
                };
            }
        })
        .collect();

    if as_ref_str_asserts.is_empty() {
        return TokenStream::new();
    }

    quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#as_ref_str_asserts)*
        }
    }
}
