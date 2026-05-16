use crate::ir::{ColumnIR, NestedLeaf, PrimitiveLeaf, TerminalLeafRoute};
use proc_macro2::TokenStream;
use quote::quote;

use super::encoder::struct_type_tokens;

fn nested_type_path(nested: NestedLeaf<'_>) -> TokenStream {
    match nested {
        NestedLeaf::Struct(ty) => struct_type_tokens(ty),
        NestedLeaf::Generic(id) => quote! { #id },
    }
}

fn column_full_dtype(
    leaf: PrimitiveLeaf<'_>,
    vec_depth: usize,
    config: &super::MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let elem_dtype = leaf.dtype(&config.external_paths);
    super::external_paths::wrap_list_layers_compile_time(pp, elem_dtype, vec_depth)
}

pub fn build_schema_entries(column: &ColumnIR, config: &super::MacroConfig) -> TokenStream {
    let name = column.name();
    match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            super::schema_nested::generate_schema_entries_for_struct(
                &type_path,
                &config.traits.to_dataframe,
                name,
                column.vec_depth(),
                &config.external_paths,
            )
        }
        TerminalLeafRoute::Primitive(leaf) => {
            let dtype = column_full_dtype(leaf, column.vec_depth(), config);
            quote! { ::std::vec![(::std::string::String::from(#name), #dtype)] }
        }
    }
}

pub fn build_empty_series(column: &ColumnIR, config: &super::MacroConfig) -> TokenStream {
    let name = column.name();
    match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            super::schema_nested::nested_empty_series_row(
                &type_path,
                &config.traits.to_dataframe,
                name,
                column.vec_depth(),
                &config.external_paths,
            )
        }
        TerminalLeafRoute::Primitive(leaf) => {
            let dtype = column_full_dtype(leaf, column.vec_depth(), config);
            let pp = config.external_paths.prelude();
            quote! { ::std::vec![#pp::Series::new_empty(#name.into(), &#dtype).into()] }
        }
    }
}
