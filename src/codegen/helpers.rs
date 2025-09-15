// RowWiseGenerator used via fully-qualified path in method references
use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_helpers_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let to_df_trait = &config.to_dataframe_trait_path;
    let it_ident = format_ident!("__df_derive_it");

    let collect_vec_impl = quote! {
            #[doc(hidden)]
            pub fn __df_derive_collect_vec_as_prefixed_list_series(items: &[#struct_name], column_name: &str) -> polars::prelude::PolarsResult<::std::vec::Vec<polars::prelude::Column>> {
                use polars::prelude::*;
                use polars::prelude::NamedFrom;
                if items.is_empty() {
                    let mut columns: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::new();
                    for (inner_name, inner_dtype) in <#struct_name as #to_df_trait>::schema()? {
                        let prefixed = format!("{}.{}", column_name, inner_name);
                        let inner_empty = Series::new_empty("".into(), &inner_dtype);
                        let list_val = AnyValue::List(inner_empty);
                        let s = Series::new(prefixed.as_str().into(), &[list_val]);
                        columns.push(s.into());
                    }
                    return Ok(columns);
                }

                let values: ::std::vec::Vec<AnyValue> = #struct_name::__df_derive_vec_to_inner_list_values(items)?;
                let schema = <#struct_name as #to_df_trait>::schema()?;
                let mut nested_series: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::with_capacity(schema.len());
                let mut iter = values.into_iter();
                for (col_name, _dtype) in schema.into_iter() {
                    let prefixed_name = format!("{}.{}", column_name, col_name);
                    let list_val = iter.next().expect("values length must match schema");
                    let list_series = Series::new(prefixed_name.as_str().into(), &[list_val]);
                    nested_series.push(list_series.into());
                }
                Ok(nested_series)
        }
    };

    let (vec_values_decls, vec_values_per_item, vec_values_finishers) =
        super::common::prepare_vec_anyvalues_parts(ir, &it_ident);

    let to_anyvalues_pieces: Vec<TokenStream> = super::strategy::build_strategies(ir)
        .iter()
        .map(super::strategy::RowWiseGenerator::gen_anyvalue_conversion)
        .collect();

    quote! {
        impl #struct_name {
            #collect_vec_impl
            #[doc(hidden)]
            pub fn __df_derive_vec_to_inner_list_values(items: &[#struct_name]) -> polars::prelude::PolarsResult<::std::vec::Vec<polars::prelude::AnyValue>> {
                use polars::prelude::*;
                if items.is_empty() {
                    let mut out_values: ::std::vec::Vec<AnyValue> = ::std::vec::Vec::new();
                    for (_inner_name, inner_dtype) in <#struct_name as #to_df_trait>::schema()? {
                        let inner_empty = Series::new_empty("".into(), &inner_dtype);
                        out_values.push(AnyValue::List(inner_empty));
                    }
                    return Ok(out_values);
                }
                #(#vec_values_decls)*
                for #it_ident in items { #(#vec_values_per_item)* }
                let mut out_values: ::std::vec::Vec<AnyValue> = ::std::vec::Vec::new();
                #(#vec_values_finishers)*
                Ok(out_values)
            }
            #[doc(hidden)]
            pub fn __df_derive_to_anyvalues(&self) -> polars::prelude::PolarsResult<::std::vec::Vec<polars::prelude::AnyValue>> {
                use polars::prelude::*;
                let mut values: ::std::vec::Vec<AnyValue> = ::std::vec::Vec::new();
                #(#to_anyvalues_pieces)*
                Ok(values)
            }
        }
    }
}
