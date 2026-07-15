use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::DeriveInput;

use crate::attrs;

use super::external_paths;

/// Runtime trait and support-type paths used by generated code.
pub struct RuntimeSurfacePaths {
    /// Fully-qualified path to the `Columnar` trait.
    pub columnar: syn::Path,
    /// Fully-qualified path to the hidden `RowCursor` trait used by generated
    /// encoders that replay source rows.
    pub row_cursor: syn::Path,
    /// Fully-qualified path to the hidden `ColumnarSpec` trait implemented by
    /// generated code.
    pub columnar_spec: syn::Path,
    /// Fully-qualified path to the hidden `ColumnSink` checked-composition
    /// boundary used by generated `ColumnarSpec` implementations.
    pub column_sink: syn::Path,
    /// Fully-qualified path to the `Decimal128Encode` trait used by Decimal
    /// fields.
    pub decimal128_encode: syn::Path,
}

/// Macro-wide configuration for generated code
#[allow(clippy::struct_field_names)]
pub struct MacroConfig {
    /// Runtime surface paths used by generated code.
    pub runtime: RuntimeSurfacePaths,
    /// External runtime dependency roots (`polars::prelude`,
    /// `polars_arrow`) used by generated code.
    pub external_paths: external_paths::ExternalPaths,
}

fn resolve_dataframe_mod_for_crate(name: &str, lib_crate_name: &str) -> Option<TokenStream> {
    match crate_name(name) {
        Ok(FoundCrate::Name(resolved)) => {
            let ident = format_ident!("{}", resolved);
            Some(quote! { ::#ident::dataframe })
        }
        Ok(FoundCrate::Itself) if is_expanding_lib_target(lib_crate_name) => {
            Some(quote! { crate::dataframe })
        }
        Ok(FoundCrate::Itself) => {
            let ident = format_ident!("{}", lib_crate_name);
            Some(quote! { ::#ident::dataframe })
        }
        Err(_) => None,
    }
}

fn is_expanding_lib_target(lib_crate_name: &str) -> bool {
    // `proc_macro_crate` reports `Itself` for every target in a package.
    // Only the library target has `crate::dataframe`; package examples,
    // benches, and integration tests need the path through the library crate.
    std::env::var("CARGO_CRATE_NAME").as_deref() == Ok(lib_crate_name)
}

pub fn resolve_default_dataframe_mod() -> TokenStream {
    // Default discovery order:
    // - `df-derive` facade (`df_derive::dataframe`, or `crate::dataframe` inside the facade)
    // - `df-derive-core` shared runtime (`df_derive_core::dataframe`)
    // - `paft-utils` direct runtime (`paft_utils::dataframe`)
    // - `paft` facade (`paft::dataframe`)
    // - local fallback (`crate::core::dataframe`)
    resolve_dataframe_mod_for_crate("df-derive", "df_derive")
        .or_else(|| resolve_dataframe_mod_for_crate("df-derive-core", "df_derive_core"))
        .or_else(|| resolve_dataframe_mod_for_crate("paft-utils", "paft_utils"))
        .or_else(|| resolve_dataframe_mod_for_crate("paft", "paft"))
        .unwrap_or_else(|| quote! { crate::core::dataframe })
}

pub fn build_macro_config(ast: &DeriveInput) -> syn::Result<MacroConfig> {
    let default_df_mod = resolve_default_dataframe_mod();
    let attrs = attrs::parse_container_attrs(ast)?;

    let uses_default_dataframe_runtime = attrs.to_dataframe.is_none() && attrs.columnar.is_none();
    let explicit_default_dataframe_mod = attrs::explicit_builtin_default_dataframe_mod(
        attrs.to_dataframe.as_ref(),
        attrs.columnar.as_ref(),
    );
    let columnar = match (&attrs.columnar, &attrs.to_dataframe) {
        (Some(override_), _) => override_.value.clone(),
        (None, Some(override_)) => attrs::rebase_last_segment(&override_.value, "Columnar"),
        (None, None) => attrs::runtime_trait_path(&default_df_mod, "Columnar"),
    };
    let row_cursor = attrs::rebase_last_segment(&columnar, "RowCursor");
    let columnar_spec = attrs::rebase_last_segment(&columnar, "ColumnarSpec");
    let column_sink = attrs::rebase_last_segment(&columnar, "ColumnSink");
    let decimal128_encode = attrs.decimal128_encode.as_ref().map_or_else(
        || attrs::rebase_last_segment(&columnar, "Decimal128Encode"),
        |override_| override_.value.clone(),
    );
    let external_paths = explicit_default_dataframe_mod.as_ref().map_or_else(
        || {
            if uses_default_dataframe_runtime {
                external_paths::default_runtime_paths(&default_df_mod)
            } else {
                external_paths::direct_dependency_paths()
            }
        },
        |dataframe_mod| {
            let dataframe_mod = quote! { #dataframe_mod };
            external_paths::default_runtime_paths(&dataframe_mod)
        },
    );

    Ok(MacroConfig {
        runtime: RuntimeSurfacePaths {
            columnar,
            row_cursor,
            columnar_spec,
            column_sink,
            decimal128_encode,
        },
        external_paths,
    })
}

#[cfg(test)]
mod tests {
    use quote::ToTokens;

    use super::*;

    #[test]
    fn standalone_columnar_override_selects_sibling_runtime_items() {
        let input: DeriveInput = syn::parse_quote! {
            #[df_derive(columnar = "custom_runtime::Columnar")]
            struct Row {
                value: CustomDecimal,
            }
        };

        let config = build_macro_config(&input).expect("runtime override should parse");

        assert_eq!(
            config.runtime.row_cursor.to_token_stream().to_string(),
            "custom_runtime :: RowCursor",
        );
        assert_eq!(
            config.runtime.columnar_spec.to_token_stream().to_string(),
            "custom_runtime :: ColumnarSpec",
        );
        assert_eq!(
            config.runtime.column_sink.to_token_stream().to_string(),
            "custom_runtime :: ColumnSink",
        );
        assert_eq!(
            config
                .runtime
                .decimal128_encode
                .to_token_stream()
                .to_string(),
            "custom_runtime :: Decimal128Encode",
        );
    }
}
