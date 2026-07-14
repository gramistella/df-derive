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

/// Walk every column, build its [`ColumnEmit`](super::column_emit::ColumnEmit),
/// and concatenate decls/pushes/builders into the three buckets the
/// generated encoder splices into `ColumnarSpec::encode_columns`. Each `ColumnEmit`
/// explicitly declares whether it contributes row-wise work or builds whole
/// columns after the loop. Concatenation is order-preserving.
fn prepare_encode_parts(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
    rows: &syn::Ident,
    sink: &syn::Ident,
) -> EncodeParts {
    let mut parts = EncodeParts::default();
    for (idx, column) in ir.columns.iter().enumerate() {
        let emit = super::column_emit::build_column_emit(column, config, idx, it_ident, rows, sink);
        match emit {
            super::column_emit::ColumnEmit::RowWise {
                decls: emit_decls,
                push,
                builders: emit_builders,
            } => {
                parts.decls.extend(emit_decls);
                parts.pushes.push(push);
                parts.builders.extend(emit_builders);
            }
            super::column_emit::ColumnEmit::WholeColumn {
                builders: emit_builders,
            } => {
                parts.builders.extend(emit_builders);
            }
        }
    }
    parts
}

fn encode_columns_method_body(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
    rows: &syn::Ident,
    sink: &syn::Ident,
) -> TokenStream {
    let EncodeParts {
        decls,
        pushes,
        builders,
    } = prepare_encode_parts(ir, config, it_ident, rows, sink);
    let push_loop = if pushes.is_empty() {
        TokenStream::new()
    } else {
        quote! { for #it_ident in #rows.iter().copied() { #(#pushes)* } }
    };

    quote! {
        #(#decls)*
        #push_loop
        #(#builders)*
        ::std::result::Result::Ok(())
    }
}

fn build_schema_method_body(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let pp = config.external_paths.prelude();
    let fields = idents::schema_fields(&ir.generics);
    let duplicate_name = idents::schema_duplicate_name(&ir.generics);
    let column_count = ir.columns.len();
    let schema_entries: Vec<TokenStream> = ir
        .columns
        .iter()
        .map(|column| super::schema::build_schema_entries(column, ir, config))
        .collect();

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
    let sink = idents::column_sink_param(&ir.generics);
    let (impl_generics, ty_generics, where_clause) =
        super::bounds::impl_parts_with_bounds(ir, config);

    let schema_body = build_schema_method_body(ir, config);
    let encode_columns_body = encode_columns_method_body(ir, config, &it_ident, &rows, &sink);

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
                let #rows: ::std::vec::Vec<&#row_lifetime Self> =
                    ::core::iter::Iterator::collect(#rows.by_ref());
                #encode_columns_body
            }
        }
    }
}
