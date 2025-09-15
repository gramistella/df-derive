use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
// trait import removed; now using common::prepare_columnar_parts

pub fn generate_columnar_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let columnar_trait = &config.columnar_trait_path;
    let to_df_trait = &config.to_dataframe_trait_path;
    let it_ident = format_ident!("__df_derive_it");

    let (decls, pushes, builders) = super::common::prepare_columnar_parts(ir, &it_ident);

    quote! {
        impl #columnar_trait for #struct_name {
            fn columnar_to_dataframe(items: &[Self]) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                if items.is_empty() {
                    return <#struct_name as #to_df_trait>::empty_dataframe();
                }
                use polars::prelude::*;
                #(#decls)*
                for #it_ident in items { #(#pushes)* }
                let mut columns: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::new();
                #(#builders)*
                if columns.is_empty() {
                    let num_rows = items.len();
                    let dummy = Series::new_empty("_dummy".into(), &DataType::Null)
                        .extend_constant(AnyValue::Null, num_rows)?;
                    let mut df = DataFrame::new(vec![dummy.into()])?;
                    df.drop_in_place("_dummy")?;
                    return Ok(df);
                }
                DataFrame::new(columns)
            }
        }
    }
}
