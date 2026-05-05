// Schema/empty-frame helpers used by `to_dataframe::schema` and
// `empty_dataframe`. Every per-row vec-anyvalues codegen path was retired
// in step 5; nested-struct columnar paths route through the encoder fold
// in the `encoder` module.

use crate::ir::{Wrapper, vec_count};
use proc_macro2::TokenStream;
use quote::quote;

use super::encoder::idents;

/// Emit a runtime loop that wraps the per-iteration `DataType` accumulator
/// (named via [`idents::schema_wrapped_dtype`]) in `layers` `List<>` envelopes.
/// Returns an empty token stream when `layers == 0` so the caller does not
/// emit `for _ in 0..0`, which trips `clippy::reversed_empty_ranges` inside
/// the user's expanded code.
fn gen_wrap_dtype_layers(layers: usize) -> TokenStream {
    if layers == 0 {
        TokenStream::new()
    } else {
        let pp = super::polars_paths::prelude();
        let wrapped = idents::schema_wrapped_dtype();
        quote! {
            for _ in 0..#layers {
                #wrapped = #pp::DataType::List(
                    ::std::boxed::Box::new(#wrapped),
                );
            }
        }
    }
}

pub fn nested_empty_series_row(
    type_path: &TokenStream,
    name: &str,
    wrappers: &[Wrapper],
) -> TokenStream {
    generate_empty_series_for_struct(type_path, name, vec_count(wrappers))
}

// --- Schema and series-shape helpers ---

pub fn generate_schema_entries_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    list_layers: usize,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let wrap_layers = gen_wrap_dtype_layers(list_layers);
    let wrapped = idents::schema_wrapped_dtype();
    quote! {
        {
            let mut nested_fields: ::std::vec::Vec<(::std::string::String, #pp::DataType)> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = ::std::format!("{}.{}", #column_name, inner_name);
                let mut #wrapped: #pp::DataType = inner_dtype;
                #wrap_layers
                nested_fields.push((prefixed_name, #wrapped));
            }
            nested_fields
        }
    }
}

fn generate_empty_series_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    list_layers: usize,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let wrap_layers = gen_wrap_dtype_layers(list_layers);
    let wrapped = idents::schema_wrapped_dtype();
    quote! {
        {
            let mut nested_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = ::std::format!("{}.{}", #column_name, inner_name);
                let mut #wrapped: #pp::DataType = inner_dtype;
                #wrap_layers
                let empty_series = #pp::Series::new_empty(prefixed_name.as_str().into(), &#wrapped);
                nested_series.push(empty_series.into());
            }
            nested_series
        }
    }
}
