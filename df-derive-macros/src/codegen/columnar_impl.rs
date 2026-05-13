use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

use super::encoder::idents;

/// Walk every field, build its [`FieldEmit`](super::strategy::FieldEmit),
/// and concatenate decls/pushes/builders into the three buckets the
/// columnar pipeline splices into the generated impl. Each `FieldEmit`
/// already places its work in the right slot — vec-bearing primitive
/// fields and every nested-struct field route their entire emit through
/// `builders` (post-loop), while leaf primitive fields contribute to all
/// three slots. Concatenation is order-preserving.
fn prepare_columnar_parts(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
) -> (Vec<TokenStream>, Vec<TokenStream>, Vec<TokenStream>) {
    let mut decls: Vec<TokenStream> = Vec::new();
    let mut pushes: Vec<TokenStream> = Vec::new();
    let mut builders: Vec<TokenStream> = Vec::new();
    for (idx, f) in ir.fields.iter().enumerate() {
        let emit = super::strategy::build_field_emit(f, config, idx, it_ident);
        decls.extend(emit.decls);
        if !emit.push.is_empty() {
            pushes.push(emit.push);
        }
        builders.extend(emit.builders);
    }
    (decls, pushes, builders)
}

fn columnar_method_body(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
) -> TokenStream {
    let to_df_trait = &config.to_dataframe_trait_path;
    let pp = config.external_paths.prelude();
    let (decls, pushes, builders) = prepare_columnar_parts(ir, config, it_ident);

    quote! {
        if items.is_empty() {
            return <Self as #to_df_trait>::empty_dataframe();
        }
        #(#decls)*
        for #it_ident in items { #(#pushes)* }
        let mut columns: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
        #(#builders)*
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
}

/// Generates the `Columnar` trait impl. The derive overrides both
/// `columnar_to_dataframe` for direct top-level `&[Self]` slices and
/// `columnar_from_refs` for borrowed nested/generic composition.
pub fn generate_columnar_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let columnar_trait = &config.columnar_trait_path;
    let pp = config.external_paths.prelude();
    let it_ident = idents::populator_iter();
    let (impl_generics, ty_generics, where_clause) = super::impl_parts_with_bounds(ir, config);

    // The method body is intentionally token-identical for `&[Self]` and
    // `&[&Self]`; field access in the borrowed path relies on Rust's
    // autoderef. Keep both trait entry points so direct slices avoid the
    // top-level `Vec<&Self>` allocation while nested emitters can compose
    // borrowed rows without cloning.
    let columnar_body = columnar_method_body(ir, config, &it_ident);
    let direct_body = columnar_body.clone();
    let refs_body = columnar_body;

    quote! {
        #[automatically_derived]
        impl #impl_generics #columnar_trait for #struct_name #ty_generics #where_clause {
            fn columnar_to_dataframe(items: &[Self]) -> #pp::PolarsResult<#pp::DataFrame> {
                #direct_body
            }

            fn columnar_from_refs(items: &[&Self]) -> #pp::PolarsResult<#pp::DataFrame> {
                #refs_body
            }
        }
    }
}
