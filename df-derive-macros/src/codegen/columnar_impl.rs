use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

use super::encoder::idents;

#[derive(Default)]
struct ColumnarParts {
    decls: Vec<TokenStream>,
    pushes: Vec<TokenStream>,
    builders: Vec<TokenStream>,
}

/// Walk every column, build its [`ColumnEmit`](super::column_emit::ColumnEmit),
/// and concatenate decls/pushes/builders into the three buckets the
/// columnar pipeline splices into the generated impl. Each `ColumnEmit`
/// explicitly declares whether it contributes row-wise work or builds whole
/// columns after the loop. Concatenation is order-preserving.
fn prepare_columnar_parts(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
    rows: &syn::Ident,
) -> ColumnarParts {
    let mut parts = ColumnarParts::default();
    for (idx, column) in ir.columns.iter().enumerate() {
        let emit = super::column_emit::build_column_emit(column, config, idx, it_ident, rows);
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

fn columnar_method_body(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
    rows: &syn::Ident,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let ColumnarParts {
        decls,
        pushes,
        builders,
    } = prepare_columnar_parts(ir, config, it_ident, rows);
    let columns = idents::columns();
    if ir.columns.is_empty() {
        return quote! {
            ::std::result::Result::Ok(
                #pp::DataFrame::empty_with_height(#rows.len()),
            )
        };
    }
    let push_loop = if pushes.is_empty() {
        TokenStream::new()
    } else {
        quote! { for #it_ident in #rows.iter().copied() { #(#pushes)* } }
    };
    let unique_name_validation = if super::support::needs_unique_name_validation(ir) {
        let validate_unique_column_names = idents::validate_unique_column_names();
        quote! {
            #validate_unique_column_names(
                #columns.iter().map(|column| column.name().as_str()),
                ::core::any::type_name::<Self>(),
            )?;
        }
    } else {
        TokenStream::new()
    };

    quote! {
        #(#decls)*
        #push_loop
        let mut #columns: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
        #(#builders)*
        #unique_name_validation
        #pp::DataFrame::new(#rows.len(), #columns)
    }
}

/// Generates the sole runtime encoding primitive, `Columnar::encode`.
pub fn generate_columnar_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let columnar_trait = &config.traits.columnar;
    let pp = config.external_paths.prelude();
    let it_ident = idents::populator_iter();
    let row_iter_param = idents::row_iter_param(&ir.generics);
    let row_lifetime = idents::row_lifetime(&ir.generics);
    let rows = idents::rows_param(&ir.generics);
    let (impl_generics, ty_generics, where_clause) =
        super::bounds::impl_parts_with_bounds(ir, config);

    let columnar_body = columnar_method_body(ir, config, &it_ident, &rows);

    quote! {
        #[automatically_derived]
        impl #impl_generics #columnar_trait for #struct_name #ty_generics #where_clause {
            fn encode<#row_lifetime, #row_iter_param>(
                #rows: #row_iter_param,
            ) -> #pp::PolarsResult<#pp::DataFrame>
            where
                Self: #row_lifetime,
                #row_iter_param: ::core::iter::IntoIterator<Item = &#row_lifetime Self>,
            {
                let #rows: ::std::vec::Vec<&#row_lifetime Self> =
                    ::core::iter::IntoIterator::into_iter(#rows).collect();
                #columnar_body
            }
        }
    }
}
