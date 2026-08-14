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
) -> ColumnarParts {
    let mut parts = ColumnarParts::default();
    for (idx, column) in ir.columns.iter().enumerate() {
        let emit = super::column_emit::build_column_emit(column, config, idx, it_ident);
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
) -> TokenStream {
    let to_df_trait = &config.traits.to_dataframe;
    let pp = config.external_paths.prelude();
    let ColumnarParts {
        decls,
        pushes,
        builders,
    } = prepare_columnar_parts(ir, config, it_ident);
    let columns = idents::columns();
    let push_loop = if pushes.is_empty() {
        TokenStream::new()
    } else {
        quote! { for #it_ident in items { #(#pushes)* } }
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
        if items.is_empty() {
            return <Self as #to_df_trait>::empty_dataframe();
        }
        #(#decls)*
        #push_loop
        let mut #columns: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
        #(#builders)*
        #unique_name_validation
        if #columns.is_empty() {
            return ::std::result::Result::Ok(
                #pp::DataFrame::empty_with_height(items.len())
            );
        }
        #pp::DataFrame::new(items.len(), #columns)
    }
}

/// Generates the `Columnar` trait impl. The derive overrides both
/// `columnar_to_dataframe` for direct top-level `&[Self]` slices and
/// `columnar_from_refs` for borrowed nested/generic composition.
pub fn generate_columnar_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let columnar_trait = &config.traits.columnar;
    let pp = config.external_paths.prelude();
    let it_ident = idents::populator_iter();
    let (impl_generics, ty_generics, where_clause) =
        super::bounds::impl_parts_with_bounds(ir, config);

    // The method body is intentionally token-identical for `&[Self]` and
    // `&[&Self]`; generated column access in the borrowed path relies on Rust's
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
