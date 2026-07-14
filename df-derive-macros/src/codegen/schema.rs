use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::{ColumnIR, NestedLeaf, PrimitiveLeaf, StructIR, TerminalLeafRoute};

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
    let element_dtype = leaf.dtype(&config.external_paths);
    super::external_paths::wrap_list_layers_compile_time(pp, element_dtype, vec_depth)
}

/// Build the ordered schema entries contributed by one terminal column.
///
/// Primitive leaves contribute one entry. Nested leaves compose their
/// explicitly declared `ColumnarSpec` schema without encoding an empty batch.
pub fn build_schema_entries(
    column: &ColumnIR,
    ir: &StructIR,
    config: &super::MacroConfig,
) -> TokenStream {
    let name = column.name();
    match column.leaf_spec().route() {
        TerminalLeafRoute::Nested(nested) => {
            let type_path = nested_type_path(nested);
            super::schema_nested::generate_schema_entries_for_struct(
                &type_path,
                &config.runtime.columnar_spec,
                name,
                column.nested_name_policy(),
                column.vec_depth(),
                &ir.generics,
                &config.external_paths,
            )
        }
        TerminalLeafRoute::Primitive(leaf) => {
            let dtype = column_full_dtype(leaf, column.vec_depth(), config);
            quote! { [(#name.into(), #dtype)] }
        }
    }
}
