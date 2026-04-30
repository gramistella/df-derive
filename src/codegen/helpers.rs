// RowWiseGenerator used via fully-qualified path in method references
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

    let collect_vec_impl = quote! {
            #[doc(hidden)]
            pub fn __df_derive_collect_vec_as_prefixed_list_series(items: &[Self], column_name: &str) -> polars::prelude::PolarsResult<::std::vec::Vec<polars::prelude::Column>> {
                use polars::prelude::*;
                use polars::prelude::NamedFrom;
                if items.is_empty() {
                    let mut columns: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::new();
                    for (inner_name, inner_dtype) in <Self as #to_df_trait>::schema()? {
                        let prefixed = format!("{}.{}", column_name, inner_name);
                        let inner_empty = Series::new_empty("".into(), &inner_dtype);
                        let list_val = AnyValue::List(inner_empty);
                        let s = Series::new(prefixed.as_str().into(), &[list_val]);
                        columns.push(s.into());
                    }
                    return Ok(columns);
                }

                let values: ::std::vec::Vec<AnyValue> = Self::__df_derive_vec_to_inner_list_values(items)?;
                let schema = <Self as #to_df_trait>::schema()?;
                let mut nested_series: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::with_capacity(schema.len());
                let mut iter = values.into_iter();
                for (col_name, _dtype) in schema.into_iter() {
                    let prefixed_name = format!("{}.{}", column_name, col_name);
                    let list_val = iter.next().expect("values length must match schema");
                    let list_series = Series::new(prefixed_name.as_str().into(), &[list_val]);
                    nested_series.push(list_series.into());
                }
                Ok(nested_series)
        }
    };

    let (vec_values_decls, vec_values_per_item, vec_values_finishers) =
        super::common::prepare_vec_anyvalues_parts(ir, &it_ident);

    let to_anyvalues_pieces: Vec<TokenStream> = super::strategy::build_strategies(ir)
        .iter()
        .map(super::strategy::RowWiseGenerator::gen_anyvalue_conversion)
        .collect();

    quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#as_ref_str_asserts)*
            #collect_vec_impl
            #[doc(hidden)]
            pub fn __df_derive_vec_to_inner_list_values(items: &[Self]) -> polars::prelude::PolarsResult<::std::vec::Vec<polars::prelude::AnyValue>> {
                use polars::prelude::*;
                if items.is_empty() {
                    let mut out_values: ::std::vec::Vec<AnyValue> = ::std::vec::Vec::new();
                    for (_inner_name, inner_dtype) in <Self as #to_df_trait>::schema()? {
                        let inner_empty = Series::new_empty("".into(), &inner_dtype);
                        out_values.push(AnyValue::List(inner_empty));
                    }
                    return Ok(out_values);
                }
                #(#vec_values_decls)*
                for #it_ident in items { #(#vec_values_per_item)* }
                let mut out_values: ::std::vec::Vec<AnyValue> = ::std::vec::Vec::new();
                #(#vec_values_finishers)*
                Ok(out_values)
            }
            #[doc(hidden)]
            pub fn __df_derive_to_anyvalues(&self) -> polars::prelude::PolarsResult<::std::vec::Vec<polars::prelude::AnyValue>> {
                use polars::prelude::*;
                let mut values: ::std::vec::Vec<AnyValue> = ::std::vec::Vec::new();
                #(#to_anyvalues_pieces)*
                Ok(values)
            }
        }
    }
}
