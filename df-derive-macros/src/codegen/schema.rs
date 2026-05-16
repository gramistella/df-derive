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
    column: &ColumnIR,
    config: &super::MacroConfig,
) -> TokenStream {
    super::type_registry::full_dtype(leaf, &column.wrapper_shape, &config.external_paths)
}

pub fn build_schema_entries(column: &ColumnIR, config: &super::MacroConfig) -> TokenStream {
    let name = column.name.as_str();
    match column.leaf_spec.route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            super::schema_nested::generate_schema_entries_for_struct(
                &type_path,
                &config.traits.to_dataframe,
                name,
                column.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        TerminalLeafRoute::Primitive(leaf) => {
            let dtype = column_full_dtype(leaf, column, config);
            quote! { ::std::vec![(::std::string::String::from(#name), #dtype)] }
        }
    }
}

pub fn build_empty_series(column: &ColumnIR, config: &super::MacroConfig) -> TokenStream {
    let name = column.name.as_str();
    match column.leaf_spec.route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            super::schema_nested::nested_empty_series_row(
                &type_path,
                &config.traits.to_dataframe,
                name,
                column.wrapper_shape.vec_depth(),
                &config.external_paths,
            )
        }
        TerminalLeafRoute::Primitive(leaf) => {
            let dtype = column_full_dtype(leaf, column, config);
            let pp = config.external_paths.prelude();
            quote! { ::std::vec![#pp::Series::new_empty(#name.into(), &#dtype).into()] }
        }
    }
}
