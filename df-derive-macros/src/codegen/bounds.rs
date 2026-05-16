use super::{MacroConfig, type_deps};
use crate::ir::{DecimalBackend, DisplayBase, LeafSpec, StringyBase, StructIR};
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Default)]
struct GenericRequirements {
    nested_params: Vec<syn::Ident>,
    nested_types: Vec<syn::Type>,
    decimal_params: Vec<syn::Ident>,
    decimal_types: Vec<syn::Type>,
    as_ref_str: Vec<syn::Ident>,
    as_ref_str_types: Vec<syn::Type>,
    display_params: Vec<syn::Ident>,
    display_types: Vec<syn::Type>,
}

fn push_unique(out: &mut Vec<syn::Ident>, ident: &syn::Ident) {
    if !out.iter().any(|existing| existing == ident) {
        out.push(ident.clone());
    }
}

fn contains_ident(items: &[syn::Ident], ident: &syn::Ident) -> bool {
    items.iter().any(|item| item == ident)
}

fn collect_leaf_requirements(leaf: &LeafSpec, reqs: &mut GenericRequirements) {
    leaf.walk_terminal_leaves(&mut |leaf| {
        if let LeafSpec::Generic(ident) = leaf {
            push_unique(&mut reqs.nested_params, ident);
        } else if let LeafSpec::Struct(ty) = leaf {
            type_deps::push_unique_type(&mut reqs.nested_types, ty);
        } else if let LeafSpec::AsStr(StringyBase::Generic(ident)) = leaf {
            push_unique(&mut reqs.as_ref_str, ident);
        } else if let LeafSpec::AsStr(StringyBase::Struct(ty)) = leaf {
            type_deps::push_unique_type(&mut reqs.as_ref_str_types, ty);
        } else if let LeafSpec::AsString(DisplayBase::Generic(ident)) = leaf {
            push_unique(&mut reqs.display_params, ident);
        } else if let LeafSpec::AsString(DisplayBase::Struct(ty)) = leaf {
            type_deps::push_unique_type(&mut reqs.display_types, ty);
        } else if let LeafSpec::Decimal { backend, .. } = leaf {
            match backend {
                DecimalBackend::RuntimeKnown => {}
                DecimalBackend::Generic(ident) => push_unique(&mut reqs.decimal_params, ident),
                DecimalBackend::Struct(ty) => {
                    type_deps::push_unique_type(&mut reqs.decimal_types, ty);
                }
            }
        }
    });
}

fn collect_generic_requirements(ir: &StructIR) -> GenericRequirements {
    let mut reqs = GenericRequirements::default();

    for column in &ir.columns {
        collect_leaf_requirements(column.leaf_spec().as_leaf_spec(), &mut reqs);
    }

    reqs
}

/// Build `impl_generics`, `ty_generics`, and `where_clause` token streams
/// suitable for splicing into an `impl` header. Generic bounds are driven by
/// each parameter's role: direct generic dataframe payloads need
/// `ToDataFrame` + `Columnar`, decimal backends need `Decimal128Encode`, generic `as_str`
/// leaves need `AsRef<str>`, generic `as_string` leaves need `Display`, and
/// concrete conversion/nested types receive exact `where` predicates.
pub(in crate::codegen) fn impl_parts_with_bounds(
    ir: &StructIR,
    config: &MacroConfig,
) -> (TokenStream, TokenStream, TokenStream) {
    let mut generics = ir.generics.clone();
    let reqs = collect_generic_requirements(ir);

    let to_df_trait = &config.traits.to_dataframe;
    let columnar_trait = &config.traits.columnar;
    let decimal_trait = &config.traits.decimal128_encode;
    let to_df_bound: syn::TypeParamBound =
        syn::parse2(quote! { #to_df_trait }).expect("trait path should parse as bound");
    let columnar_bound: syn::TypeParamBound =
        syn::parse2(quote! { #columnar_trait }).expect("trait path should parse as bound");
    let decimal_bound: syn::TypeParamBound =
        syn::parse2(quote! { #decimal_trait }).expect("trait path should parse as bound");
    let as_ref_str_bound: syn::TypeParamBound = syn::parse2(quote! { ::core::convert::AsRef<str> })
        .expect("AsRef<str> should parse as bound");
    let display_bound: syn::TypeParamBound =
        syn::parse2(quote! { ::core::fmt::Display }).expect("Display should parse as bound");
    // No `Clone` bound: bulk emitters collect `Vec<&T>` and route through
    // `Columnar::columnar_from_refs`, and every primitive-vec branch in the
    // encoder IR borrows from the for-loop binding directly. A user with a
    // non-`Clone` payload (e.g. `T: ToDataFrame + Columnar` only) can derive
    // `ToDataFrame` on a struct holding `T` without that bound leaking from
    // the macro.
    for tp in generics.type_params_mut() {
        if contains_ident(&reqs.nested_params, &tp.ident) {
            tp.bounds.push(to_df_bound.clone());
            tp.bounds.push(columnar_bound.clone());
        }

        if contains_ident(&reqs.decimal_params, &tp.ident) {
            tp.bounds.push(decimal_bound.clone());
        }

        if contains_ident(&reqs.as_ref_str, &tp.ident) {
            tp.bounds.push(as_ref_str_bound.clone());
        }

        if contains_ident(&reqs.display_params, &tp.ident) {
            tp.bounds.push(display_bound.clone());
        }
    }

    if !reqs.nested_types.is_empty()
        || !reqs.as_ref_str_types.is_empty()
        || !reqs.display_types.is_empty()
        || !reqs.decimal_types.is_empty()
    {
        let where_clause_mut = generics.make_where_clause();

        for ty in &reqs.nested_types {
            let nested_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #nested_ty: #to_df_trait })
                    .expect("nested ToDataFrame where predicate should parse"),
            );
            where_clause_mut.predicates.push(
                syn::parse2(quote! { #nested_ty: #columnar_trait })
                    .expect("nested Columnar where predicate should parse"),
            );
        }
        for ty in &reqs.as_ref_str_types {
            let as_str_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #as_str_ty: ::core::convert::AsRef<str> })
                    .expect("as_str type where predicate should parse"),
            );
        }
        for ty in &reqs.display_types {
            let display_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #display_ty: ::core::fmt::Display })
                    .expect("as_string type where predicate should parse"),
            );
        }
        for ty in &reqs.decimal_types {
            let decimal_ty = quote! { #ty };

            where_clause_mut.predicates.push(
                syn::parse2(quote! { #decimal_ty: #decimal_trait })
                    .expect("decimal backend where predicate should parse"),
            );
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    (
        quote! { #impl_generics },
        quote! { #ty_generics },
        quote! { #where_clause },
    )
}
