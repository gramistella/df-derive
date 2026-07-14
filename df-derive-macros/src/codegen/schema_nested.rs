use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::NestedNamePolicy;

use super::encoder::idents;
use super::external_paths::ExternalPaths;

/// Compose the explicitly declared schema of a nested row type into its
/// parent's ordered schema entries.
pub fn generate_schema_entries_for_struct(
    type_path: &TokenStream,
    columnar_spec_trait: &syn::Path,
    column_name: &str,
    name_policy: &NestedNamePolicy,
    list_layers: usize,
    generics: &syn::Generics,
    paths: &ExternalPaths,
) -> TokenStream {
    let pp = paths.prelude();
    let nested_fields = idents::schema_nested_fields(generics);
    let inner_name = idents::schema_inner_name(generics);
    let inner_dtype = idents::schema_inner_dtype(generics);
    let output_name = idents::schema_output_name(generics);
    let wrapped = idents::schema_wrapped_dtype(generics);
    let wrap_layers = super::external_paths::wrap_list_layers_runtime(pp, &wrapped, list_layers);
    let composed_name =
        super::nested_names::compose_nested_name(name_policy, column_name, &quote! { #inner_name });

    quote! {
        {
            let mut #nested_fields: ::std::vec::Vec<(#pp::PlSmallStr, #pp::DataType)> =
                ::std::vec::Vec::new();
            for (#inner_name, #inner_dtype) in
                <#type_path as #columnar_spec_trait>::build_schema()?.iter()
            {
                let #output_name = #composed_name;
                let mut #wrapped: #pp::DataType = #inner_dtype.clone();
                #wrap_layers
                #nested_fields.push((#output_name.into(), #wrapped));
            }
            #nested_fields
        }
    }
}
