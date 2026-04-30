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

#[allow(clippy::too_many_lines)]
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

    let collect_vec_impl = quote! {
            #[doc(hidden)]
            pub fn __df_derive_collect_vec_as_prefixed_list_series(items: &[Self], column_name: &str) -> #pp::PolarsResult<::std::vec::Vec<#pp::Column>> {
                if items.is_empty() {
                    let mut columns: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
                    for (inner_name, inner_dtype) in <Self as #to_df_trait>::schema()? {
                        let prefixed = ::std::format!("{}.{}", column_name, inner_name);
                        let inner_empty = #pp::Series::new_empty("".into(), &inner_dtype);
                        let list_val = #pp::AnyValue::List(inner_empty);
                        let s = <#pp::Series as #pp::NamedFrom<_, _>>::new(
                            prefixed.as_str().into(),
                            &[list_val],
                        );
                        columns.push(s.into());
                    }
                    return ::std::result::Result::Ok(columns);
                }

                let values: ::std::vec::Vec<#pp::AnyValue> =
                    Self::__df_derive_vec_to_inner_list_values(items)?;
                let schema = <Self as #to_df_trait>::schema()?;
                let mut nested_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::with_capacity(schema.len());
                let mut iter = values.into_iter();
                for (col_name, _dtype) in schema.into_iter() {
                    let prefixed_name = ::std::format!("{}.{}", column_name, col_name);
                    let list_val = iter.next().ok_or_else(|| #pp::polars_err!(
                        ComputeError: "df-derive: __df_derive_vec_to_inner_list_values produced fewer values than schema columns (codegen invariant violation)"
                    ))?;
                    let list_series = <#pp::Series as #pp::NamedFrom<_, _>>::new(
                        prefixed_name.as_str().into(),
                        &[list_val],
                    );
                    nested_series.push(list_series.into());
                }
                ::std::result::Result::Ok(nested_series)
        }
    };

    let (vec_values_decls, vec_values_per_item, vec_values_finishers) =
        super::common::prepare_vec_anyvalues_parts(ir, config, &it_ident);

    let to_anyvalues_pieces: Vec<TokenStream> = super::strategy::build_strategies(ir, config)
        .iter()
        .map(super::strategy::Strategy::gen_anyvalue_conversion)
        .collect();

    let (cf_decls, cf_pushes, cf_builders) =
        super::common::prepare_columnar_parts(ir, config, &it_ident);

    quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#as_ref_str_asserts)*
            #collect_vec_impl
            /// Builds the columnar `DataFrame` for a slice of references to
            /// `Self`. Used both by the trait `columnar_to_dataframe` shim
            /// (which collects refs from `&[Self]`) and by the bulk-vec
            /// emitter on parent structs that hold a `Vec<Self>` field
            /// (which flattens a parent's inner Vecs into a `Vec<&Self>`
            /// without requiring `Self: Clone`).
            #[doc(hidden)]
            pub fn __df_derive_columnar_from_refs(items: &[&Self]) -> #pp::PolarsResult<#pp::DataFrame> {
                if items.is_empty() {
                    return <Self as #to_df_trait>::empty_dataframe();
                }
                #(#cf_decls)*
                for #it_ident in items { #(#cf_pushes)* }
                let mut columns: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
                #(#cf_builders)*
                if columns.is_empty() {
                    let num_rows = items.len();
                    let dummy = #pp::Series::new_empty(
                        "_dummy".into(),
                        &#pp::DataType::Null,
                    )
                    .extend_constant(#pp::AnyValue::Null, num_rows)?;
                    let mut df = #pp::DataFrame::new_infer_height(::std::vec![dummy.into()])?;
                    df.drop_in_place("_dummy")?;
                    return ::std::result::Result::Ok(df);
                }
                #pp::DataFrame::new_infer_height(columns)
            }
            #[doc(hidden)]
            pub fn __df_derive_vec_to_inner_list_values(items: &[Self]) -> #pp::PolarsResult<::std::vec::Vec<#pp::AnyValue>> {
                if items.is_empty() {
                    let mut out_values: ::std::vec::Vec<#pp::AnyValue> = ::std::vec::Vec::new();
                    for (_inner_name, inner_dtype) in <Self as #to_df_trait>::schema()? {
                        let inner_empty = #pp::Series::new_empty("".into(), &inner_dtype);
                        out_values.push(#pp::AnyValue::List(inner_empty));
                    }
                    return ::std::result::Result::Ok(out_values);
                }
                #(#vec_values_decls)*
                for #it_ident in items { #(#vec_values_per_item)* }
                let mut out_values: ::std::vec::Vec<#pp::AnyValue> = ::std::vec::Vec::new();
                #(#vec_values_finishers)*
                ::std::result::Result::Ok(out_values)
            }
            #[doc(hidden)]
            pub fn __df_derive_to_anyvalues(&self) -> #pp::PolarsResult<::std::vec::Vec<#pp::AnyValue>> {
                let mut values: ::std::vec::Vec<#pp::AnyValue> = ::std::vec::Vec::new();
                #(#to_anyvalues_pieces)*
                ::std::result::Result::Ok(values)
            }
        }
    }
}
