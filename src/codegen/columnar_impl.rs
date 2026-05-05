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

/// Generates the `Columnar` trait impl. The derive overrides
/// `columnar_from_refs` (the borrowed entry point on the trait), which is
/// where the actual columnar-build logic lives. The trait's default
/// `columnar_to_dataframe` collects refs and delegates here, so we don't
/// emit it.
///
/// Routing every entry point through `columnar_from_refs` lets parent bulk
/// emitters call `<T as Columnar>::columnar_from_refs(&refs)` for both
/// generic-parameter and concrete-struct nested fields without needing
/// `T: Clone`. The `Vec<&Self>` allocation paid by `columnar_to_dataframe`'s
/// default delegation costs N usize per call — negligible next to the
/// columnar work it does.
pub fn generate_columnar_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let columnar_trait = &config.columnar_trait_path;
    let to_df_trait = &config.to_dataframe_trait_path;
    let pp = super::polars_paths::prelude();
    let it_ident = idents::populator_iter();
    let (impl_generics, ty_generics, where_clause) =
        super::impl_parts_with_bounds(&ir.generics, config);

    let (cf_decls, cf_pushes, cf_builders) = prepare_columnar_parts(ir, config, &it_ident);

    quote! {
        impl #impl_generics #columnar_trait for #struct_name #ty_generics #where_clause {
            fn columnar_from_refs(items: &[&Self]) -> #pp::PolarsResult<#pp::DataFrame> {
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
        }
    }
}
