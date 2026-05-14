use syn::{Ident, PathArguments, Type, TypePath};

use super::known_types::is_bare_str_type;
use super::path_match::{PathView, path_prefix_is_no_args};

pub(super) fn reject_unsupported_collection_type(current_type: &Type) -> Result<(), syn::Error> {
    if let Type::Path(type_path) = current_type
        && let Some(collection) = unsupported_collection_kind(type_path)
    {
        return Err(collection.diagnostic(current_type));
    }
    Ok(())
}

pub(super) fn reject_bare_duration(
    current_type: &Type,
    generic_params: &[Ident],
) -> Result<(), syn::Error> {
    if let Type::Path(type_path) = current_type
        && type_path.qself.is_none()
        && type_path.path.segments.len() == 1
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "Duration"
        && matches!(segment.arguments, PathArguments::None)
        && !generic_params.iter().any(|p| p == &segment.ident)
    {
        return Err(syn::Error::new_spanned(
            current_type,
            "bare `Duration` is ambiguous; use `std::time::Duration`, \
             `core::time::Duration`, or `chrono::Duration` to disambiguate",
        ));
    }
    Ok(())
}

pub(super) fn reject_bare_unsized_leaf(current_type: &Type) -> Result<(), syn::Error> {
    if is_bare_str_type(current_type) {
        return Err(syn::Error::new_spanned(
            current_type,
            "df-derive does not support bare or smart-pointer-wrapped `str` leaves; \
             use `String`, `&str`, `Cow<'_, str>`, or a sized wrapper such as \
             `Box<String>`",
        ));
    }
    if matches!(current_type, Type::Slice(_)) {
        return Err(syn::Error::new_spanned(
            current_type,
            "df-derive does not support bare or smart-pointer-wrapped `[T]` slice \
             leaves; use `Vec<T>` for list columns, or use `&[u8]`/`Cow<'_, [u8]>` \
             with `#[df_derive(as_binary)]` for borrowed binary data",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum UnsupportedCollection {
    HashMap,
    BTreeMap,
    HashSet,
    BTreeSet,
    VecDeque,
    LinkedList,
}

impl UnsupportedCollection {
    const ALL: [Self; 6] = [
        Self::HashMap,
        Self::BTreeMap,
        Self::HashSet,
        Self::BTreeSet,
        Self::VecDeque,
        Self::LinkedList,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::HashMap => "HashMap",
            Self::BTreeMap => "BTreeMap",
            Self::HashSet => "HashSet",
            Self::BTreeSet => "BTreeSet",
            Self::VecDeque => "VecDeque",
            Self::LinkedList => "LinkedList",
        }
    }

    fn diagnostic(self, current_type: &Type) -> syn::Error {
        let message = match self {
            Self::HashMap => "df-derive does not support `HashMap` fields. Convert to \
                 `Vec<(K, V)>` or pre-flatten into named columns before assignment."
                .to_owned(),
            Self::BTreeMap => "df-derive does not support `BTreeMap` fields. Convert to \
                 `Vec<(K, V)>` or pre-flatten into named columns before assignment."
                .to_owned(),
            Self::HashSet => "df-derive does not support `HashSet` fields. Convert to \
                 `Vec<T>` (order will be set-defined, not insertion-defined)."
                .to_owned(),
            Self::BTreeSet => "df-derive does not support `BTreeSet` fields. Convert to \
                 `Vec<T>` (order will follow the set's sorted iteration order)."
                .to_owned(),
            Self::VecDeque | Self::LinkedList => {
                let collection = self.name();
                format!(
                    "df-derive does not support `{collection}` fields. Convert to `Vec<T>` before assignment."
                )
            }
        };
        syn::Error::new_spanned(current_type, message)
    }
}

fn unsupported_collection_kind(type_path: &TypePath) -> Option<UnsupportedCollection> {
    UnsupportedCollection::ALL
        .into_iter()
        .find(|collection| path_is_bare_or_std_collection(type_path, collection.name()))
}

fn path_is_bare_or_std_collection(type_path: &TypePath, leaf: &str) -> bool {
    let Some(path) = PathView::from_type_path(type_path) else {
        return false;
    };
    let Some(segment) = path.leaf() else {
        return false;
    };

    segment.ident == leaf
        && (path.len() == 1
            || (path.len() == 3 && path_prefix_is_no_args(type_path, &["std", "collections"])))
}
