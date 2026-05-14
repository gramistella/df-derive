use crate::codegen::MacroConfig;
use crate::codegen::strategy::EmitMode;
use crate::ir::{FieldIR, LeafRoute, TupleElement};
use proc_macro2::TokenStream;
use quote::quote;

/// Build the schema/empty-rows entries for a tuple-typed field. Each element
/// contributes one or more entries; nested tuples recurse, and nested
/// structs/generic parameters delegate to the existing schema helpers (which
/// iterate `T::schema()?` at runtime).
pub fn build_field_entries(
    field: &FieldIR,
    elements: &[TupleElement],
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    let parent_name = crate::codegen::helpers::column_name_for_ident(&field.name);
    let outer_layers = field.wrapper_shape.vec_depth();
    build_tuple_entries(elements, &parent_name, outer_layers, mode, config)
}

fn build_tuple_entries(
    elements: &[TupleElement],
    column_prefix: &str,
    outer_layers: usize,
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let mut per_elem: Vec<TokenStream> = Vec::with_capacity(elements.len());
    for (i, elem) in elements.iter().enumerate() {
        let elem_prefix = format!("{column_prefix}.field_{i}");
        per_elem.push(build_element_entries(
            elem,
            &elem_prefix,
            outer_layers,
            mode,
            config,
        ));
    }
    match mode {
        EmitMode::SchemaEntries => quote! {
            {
                let mut tuple_fields: ::std::vec::Vec<(::std::string::String, #pp::DataType)> =
                    ::std::vec::Vec::new();
                #(
                    tuple_fields.extend(#per_elem);
                )*
                tuple_fields
            }
        },
        EmitMode::EmptyRows => quote! {
            {
                let mut tuple_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
                #(
                    tuple_series.extend(#per_elem);
                )*
                tuple_series
            }
        },
    }
}

fn build_element_entries(
    elem: &TupleElement,
    column_prefix: &str,
    outer_layers: usize,
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let total_layers = outer_layers + elem.wrapper_shape.vec_depth();
    match elem.leaf_spec.route() {
        LeafRoute::Nested(nested) => {
            let type_path = super::tuple_nested_type_path(nested);
            element_nested_entries(&type_path, column_prefix, total_layers, mode, config)
        }
        LeafRoute::Tuple(inner) => {
            build_tuple_entries(inner, column_prefix, total_layers, mode, config)
        }
        LeafRoute::Primitive(leaf) => {
            let elem_dtype = leaf.dtype(&config.external_paths);
            let full_dtype = crate::codegen::external_paths::wrap_list_layers_compile_time(
                pp,
                elem_dtype,
                total_layers,
            );
            match mode {
                EmitMode::SchemaEntries => quote! {
                    ::std::vec![(::std::string::String::from(#column_prefix), #full_dtype)]
                },
                EmitMode::EmptyRows => quote! {
                    ::std::vec![
                        #pp::Series::new_empty(#column_prefix.into(), &#full_dtype).into()
                    ]
                },
            }
        }
    }
}

fn element_nested_entries(
    type_path: &TokenStream,
    column_prefix: &str,
    total_layers: usize,
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    match mode {
        EmitMode::SchemaEntries => {
            crate::codegen::schema_nested::generate_schema_entries_for_struct(
                type_path,
                &config.traits.to_dataframe,
                column_prefix,
                total_layers,
                &config.external_paths,
            )
        }
        EmitMode::EmptyRows => crate::codegen::schema_nested::nested_empty_series_row(
            type_path,
            &config.traits.to_dataframe,
            column_prefix,
            total_layers,
            &config.external_paths,
        ),
    }
}
