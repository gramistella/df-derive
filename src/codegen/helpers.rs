use crate::ir::{PrimitiveTransform, StructIR};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Peel `Option<...>` and `Vec<...>` wrappers off a field type to find the
/// leaf used by codegen. Mirrors the wrapper-stripping in
/// `type_analysis::analyze_type` so the leaf type used in the trait-bound
/// assert matches the codegen's notion of the element type.
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
        break;
    }
    cur
}

pub fn generate_helpers_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let to_df_trait = &config.to_dataframe_trait_path;
    let pp = super::polars_paths::prelude();
    let it_ident = format_ident!("__df_derive_it");
    let (impl_generics, ty_generics, where_clause) =
        super::impl_parts_with_bounds(&ir.generics, config);

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
        .filter(|(_, f)| matches!(f.transform, Some(PrimitiveTransform::AsStr)))
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

    let (vec_values_decls, vec_values_per_item, vec_values_finishers) =
        super::common::prepare_vec_anyvalues_parts(ir, config, &it_ident);

    // Per-item field access in `vec_values_per_item` is `__df_derive_it.field`,
    // which auto-derefs through any number of references — so the same body
    // works whether `__df_derive_it: &Self` or `&&Self`. The two helpers below
    // inline the same body to avoid the extra `Vec<&Self>` allocation that a
    // `vec`-as-thin-wrapper-around-`refs` adapter would impose on the
    // `Vec<Struct>` flat path.
    let body = quote! {
        if items.is_empty() {
            let mut out_values: ::std::vec::Vec<#pp::AnyValue<'static>> = ::std::vec::Vec::new();
            for (_inner_name, inner_dtype) in <Self as #to_df_trait>::schema()? {
                let inner_empty = #pp::Series::new_empty("".into(), &inner_dtype);
                out_values.push(#pp::AnyValue::List(inner_empty));
            }
            return ::std::result::Result::Ok(out_values);
        }
        #(#vec_values_decls)*
        for #it_ident in items { #(#vec_values_per_item)* }
        let mut out_values: ::std::vec::Vec<#pp::AnyValue<'static>> = ::std::vec::Vec::new();
        #(#vec_values_finishers)*
        ::std::result::Result::Ok(out_values)
    };

    quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#as_ref_str_asserts)*
            #[doc(hidden)]
            pub fn __df_derive_vec_to_inner_list_values(items: &[Self]) -> #pp::PolarsResult<::std::vec::Vec<#pp::AnyValue<'static>>> {
                #body
            }
            #[doc(hidden)]
            pub fn __df_derive_refs_to_inner_list_values(items: &[&Self]) -> #pp::PolarsResult<::std::vec::Vec<#pp::AnyValue<'static>>> {
                #body
            }
        }
    }
}
