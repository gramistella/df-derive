use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

use super::encoder::idents;

#[derive(Default)]
struct EncodeParts {
    decls: Vec<TokenStream>,
    pushes: Vec<TokenStream>,
    builders: Vec<TokenStream>,
}

/// Walk every source field and concatenate its declaration, per-row, and
/// materialization phases. Tuple fields contribute one recursive push tree,
/// regardless of how many terminal columns they contain.
fn prepare_encode_parts(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
    row_capacity: &syn::Ident,
    sink: &syn::Ident,
) -> EncodeParts {
    let mut parts = EncodeParts::default();
    let mut terminal_idx = 0;
    let mut group_idx = 0;
    let ident_scope = idents::GeneratedIdentScope::new(&ir.generics);
    for field in &ir.fields {
        let emit = super::column_emit::build_field_emit(
            field,
            super::column_emit::FieldEmitParams {
                config,
                ident_scope,
                terminal_start: terminal_idx,
                group_start: group_idx,
                row: it_ident,
                row_capacity,
                sink,
            },
        );
        terminal_idx += emit.terminal_count;
        group_idx += emit.group_count;
        parts.decls.extend(emit.decls);
        parts.pushes.push(emit.push);
        parts.builders.extend(emit.builders);
    }
    debug_assert_eq!(terminal_idx, ir.terminal_column_count());
    parts
}

fn encode_columns_method_body(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
    rows: &syn::Ident,
    row_capacity: &syn::Ident,
    sink: &syn::Ident,
) -> TokenStream {
    let ident_scope = idents::GeneratedIdentScope::new(&ir.generics);
    let row_upper_bound = idents::row_upper_bound(ident_scope);
    let input_rows_exact = idents::input_rows_exact(ident_scope);
    let EncodeParts {
        decls,
        pushes,
        builders,
    } = prepare_encode_parts(ir, config, it_ident, row_capacity, sink);

    quote! {
        let (#row_capacity, #row_upper_bound) =
            ::core::iter::Iterator::size_hint(&*#rows);
        let #input_rows_exact: bool =
            #row_upper_bound == ::std::option::Option::Some(#row_capacity);
        let _ = #input_rows_exact;
        #(#decls)*
        for #it_ident in #rows.by_ref() {
            #(#pushes)*
            let _ = #it_ident;
        }
        #(#builders)*
        ::std::result::Result::Ok(())
    }
}

fn build_schema_method_body(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let pp = config.external_paths.prelude();
    let fields = idents::schema_fields(&ir.generics);
    let duplicate_name = idents::schema_duplicate_name(&ir.generics);
    let column_count = ir.terminal_column_count();
    let mut schema_entries = Vec::with_capacity(column_count);
    ir.visit_terminal_columns(|column| {
        schema_entries.push(super::schema::build_schema_entries(column, ir, config));
    });

    quote! {
        let mut #fields: ::std::vec::Vec<(#pp::PlSmallStr, #pp::DataType)> =
            ::std::vec::Vec::with_capacity(#column_count);
        #(
            #fields.extend(#schema_entries);
        )*
        ::std::result::Result::Ok(::std::sync::Arc::new(
            #pp::Schema::try_from_iter_check_duplicates(
                #fields.into_iter().map(::std::result::Result::Ok),
                |#duplicate_name: &str| #pp::polars_err!(
                    ComputeError:
                    "df-derive: duplicate column `{}` while building schema for {}",
                    #duplicate_name,
                    ::core::any::type_name::<Self>(),
                ),
            )?,
        ))
    }
}

/// Generates the hidden schema/column specification consumed by the checked
/// runtime boundary.
pub fn generate_columnar_spec_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let columnar_spec_trait = &config.runtime.columnar_spec;
    let column_sink_type = &config.runtime.column_sink;
    let pp = config.external_paths.prelude();
    let it_ident = idents::populator_iter();
    let row_iter_param = idents::row_iter_param(&ir.generics);
    let row_lifetime = idents::row_lifetime(&ir.generics);
    let rows = idents::rows_param(&ir.generics);
    let row_capacity = idents::row_capacity(&ir.generics);
    let sink = idents::column_sink_param(&ir.generics);
    let (impl_generics, ty_generics, where_clause) =
        super::bounds::impl_parts_with_bounds(ir, config);

    let schema_body = build_schema_method_body(ir, config);
    let encode_columns_body =
        encode_columns_method_body(ir, config, &it_ident, &rows, &row_capacity, &sink);

    quote! {
        #[automatically_derived]
        impl #impl_generics #columnar_spec_trait for #struct_name #ty_generics #where_clause {
            fn build_schema() -> #pp::PolarsResult<#pp::SchemaRef> {
                #schema_body
            }

            fn encode_columns<#row_lifetime, #row_iter_param>(
                #rows: &mut #row_iter_param,
                #sink: &mut #column_sink_type,
            ) -> #pp::PolarsResult<()>
            where
                Self: #row_lifetime,
                #row_iter_param: ::core::iter::Iterator<Item = &#row_lifetime Self>,
            {
                #encode_columns_body
            }
        }
    }
}
