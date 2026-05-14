use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::{DeriveInput, PathArguments};

pub struct ContainerAttrs {
    pub to_dataframe: Option<RuntimeOverridePath>,
    pub columnar: Option<RuntimeOverridePath>,
    pub decimal128_encode: Option<RuntimeOverridePath>,
}

pub struct RuntimeOverridePath {
    pub path: syn::Path,
    pub span: Span,
}

/// Parse a `key = "path::Trait"` attribute value into a `syn::Path`, with a
/// uniform error message of the form `"invalid {label} path: {e}"`. Callers
/// pass the full noun phrase (e.g., `"trait"`, `"columnar trait"`,
/// `"decimal128_encode trait"`) so the existing user-facing strings are
/// preserved verbatim.
fn parse_trait_path_attr(
    meta: &syn::meta::ParseNestedMeta<'_>,
    label: &str,
) -> syn::Result<syn::Path> {
    let lit: syn::LitStr = meta.value()?.parse()?;
    syn::parse_str(&lit.value()).map_err(|e| meta.error(format!("invalid {label} path: {e}")))
}

/// Clone `path` and replace the last segment's identifier with `name`,
/// preserving the original span. Used to derive sibling trait paths
/// (`Columnar`, `Decimal128Encode`) from a user-supplied `ToDataFrame` path.
pub fn rebase_last_segment(path: &syn::Path, name: &str) -> syn::Path {
    let mut new_path = path.clone();
    if let Some(last_segment) = new_path.segments.last_mut() {
        last_segment.ident = syn::Ident::new(name, last_segment.ident.span());
    }
    new_path
}

fn set_runtime_override(
    slot: &mut Option<RuntimeOverridePath>,
    key: &'static str,
    path: syn::Path,
    incoming_span: Span,
) -> syn::Result<()> {
    if let Some(existing) = slot {
        let mut error = syn::Error::new(
            incoming_span,
            format!("container attribute declares duplicate `{key}` override; remove one"),
        );
        error.combine(syn::Error::new(
            existing.span,
            format!("first `{key}` override declared here"),
        ));
        return Err(error);
    }

    *slot = Some(RuntimeOverridePath {
        path,
        span: incoming_span,
    });
    Ok(())
}

fn reject_columnar_without_trait(columnar_span: Span) -> syn::Error {
    syn::Error::new(
        columnar_span,
        "`columnar = \"...\"` requires `trait = \"...\"`; overriding only \
         `Columnar` would generate mixed runtime impls that do not satisfy \
         either runtime's `ToDataFrameVec`",
    )
}

fn mixed_builtin_runtime_error(
    trait_override: &RuntimeOverridePath,
    columnar_override: &RuntimeOverridePath,
) -> syn::Error {
    let mut error = syn::Error::new(
        columnar_override.span,
        "`trait` and `columnar` overrides cannot mix the built-in dataframe \
         runtime with a custom runtime; use the matching built-in `Columnar` \
         path or provide a fully custom `trait` + `columnar` pair",
    );
    error.combine(syn::Error::new(
        trait_override.span,
        "`trait` override involved in the mixed runtime pair",
    ));
    error
}

fn path_segment_names(path: &syn::Path) -> Option<Vec<String>> {
    path.segments
        .iter()
        .map(|segment| {
            if matches!(segment.arguments, PathArguments::None) {
                Some(segment.ident.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn dataframe_mod_segments_for_crate(
    package_name: &str,
    lib_crate_name: &str,
) -> Option<[String; 2]> {
    let root = match crate_name(package_name) {
        Ok(FoundCrate::Name(resolved)) => resolved,
        Ok(FoundCrate::Itself)
            if std::env::var("CARGO_CRATE_NAME").as_deref() == Ok(lib_crate_name) =>
        {
            "crate".to_owned()
        }
        Ok(FoundCrate::Itself) => lib_crate_name.to_owned(),
        Err(_) => return None,
    };

    Some([root, "dataframe".to_owned()])
}

fn is_builtin_default_dataframe_mod(path: &syn::Path) -> bool {
    let Some(segments) = path_segment_names(path) else {
        return false;
    };

    [
        dataframe_mod_segments_for_crate("df-derive", "df_derive"),
        dataframe_mod_segments_for_crate("df-derive-core", "df_derive_core"),
    ]
    .into_iter()
    .flatten()
    .any(|expected| segments.as_slice() == expected.as_slice())
}

fn trait_module_path(path: &syn::Path, trait_name: &str) -> Option<syn::Path> {
    let last = path.segments.last()?;
    if !matches!(last.arguments, PathArguments::None) || last.ident != trait_name {
        return None;
    }

    let mut module_path = path.clone();
    let _ = module_path.segments.pop();
    let _ = module_path.segments.pop_punct();
    (!module_path.segments.is_empty()).then_some(module_path)
}

fn path_segments_equal(left: &syn::Path, right: &syn::Path) -> bool {
    path_segment_names(left) == path_segment_names(right)
}

fn mixed_builtin_runtime_override(
    to_df_trait_path: Option<&RuntimeOverridePath>,
    columnar_trait_path: Option<&RuntimeOverridePath>,
) -> Option<syn::Error> {
    let to_df_trait_path = to_df_trait_path?;
    let columnar_trait_path = columnar_trait_path?;
    let to_df_module = trait_module_path(&to_df_trait_path.path, "ToDataFrame")?;
    let columnar_module = trait_module_path(&columnar_trait_path.path, "Columnar")?;
    let to_df_builtin = is_builtin_default_dataframe_mod(&to_df_module);
    let columnar_builtin = is_builtin_default_dataframe_mod(&columnar_module);

    ((to_df_builtin || columnar_builtin) && !path_segments_equal(&to_df_module, &columnar_module))
        .then(|| mixed_builtin_runtime_error(to_df_trait_path, columnar_trait_path))
}

pub fn explicit_builtin_default_dataframe_mod(
    to_df_trait_path: Option<&RuntimeOverridePath>,
    columnar_trait_path: Option<&RuntimeOverridePath>,
) -> Option<syn::Path> {
    let to_df_module = trait_module_path(&to_df_trait_path?.path, "ToDataFrame")?;
    if !is_builtin_default_dataframe_mod(&to_df_module) {
        return None;
    }

    if let Some(columnar) = columnar_trait_path {
        let columnar_module = trait_module_path(&columnar.path, "Columnar")?;
        if !path_segments_equal(&to_df_module, &columnar_module) {
            return None;
        }
    }

    Some(to_df_module)
}

pub fn parse_container_attrs(input: &DeriveInput) -> syn::Result<ContainerAttrs> {
    let mut to_dataframe: Option<RuntimeOverridePath> = None;
    let mut columnar: Option<RuntimeOverridePath> = None;
    let mut decimal128_encode: Option<RuntimeOverridePath> = None;

    for attr in &input.attrs {
        if attr.path().is_ident("df_derive") {
            attr.parse_nested_meta(|meta| {
                let key_span = meta.path.span();
                if meta.path.is_ident("trait") {
                    let path = parse_trait_path_attr(&meta, "trait")?;
                    set_runtime_override(&mut to_dataframe, "trait", path, key_span)
                } else if meta.path.is_ident("columnar") {
                    let path = parse_trait_path_attr(&meta, "columnar trait")?;
                    set_runtime_override(&mut columnar, "columnar", path, key_span)
                } else if meta.path.is_ident("decimal128_encode") {
                    let path = parse_trait_path_attr(&meta, "decimal128_encode trait")?;
                    set_runtime_override(
                        &mut decimal128_encode,
                        "decimal128_encode",
                        path,
                        key_span,
                    )
                } else {
                    Err(meta.error("unsupported key in #[df_derive(...)] attribute"))
                }
            })?;
        }
    }

    if let (Some(columnar), None) = (&columnar, &to_dataframe) {
        return Err(reject_columnar_without_trait(columnar.span));
    }
    if let Some(err) = mixed_builtin_runtime_override(to_dataframe.as_ref(), columnar.as_ref()) {
        return Err(err);
    }

    Ok(ContainerAttrs {
        to_dataframe,
        columnar,
        decimal128_encode,
    })
}
