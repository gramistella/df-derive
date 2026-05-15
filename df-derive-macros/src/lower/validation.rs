use crate::type_analysis::{AnalyzedBase, AnalyzedType};
use syn::Ident;

use super::diagnostics;

fn is_direct_self_type(ty: &syn::Type, struct_name: &Ident) -> bool {
    let syn::Type::Path(type_path) = ty else {
        return false;
    };
    if type_path.qself.is_some() {
        return false;
    }
    let segments = &type_path.path.segments;
    let Some(segment) = segments.last() else {
        return false;
    };
    if segments.len() == 1 {
        return segment.ident == "Self" || &segment.ident == struct_name;
    }
    if segments.len() != 2
        || !segments
            .iter()
            .all(|segment| matches!(segment.arguments, syn::PathArguments::None))
    {
        return false;
    }
    let Some(first_segment) = segments.first() else {
        return false;
    };
    // Keep this intentionally narrow: broader qualified paths can name
    // distinct same-named types outside the deriving type's module.
    (first_segment.ident == "crate" || first_segment.ident == "self")
        && &segment.ident == struct_name
}

pub fn reject_direct_self_reference(
    analyzed: &AnalyzedType,
    field_display_name: &str,
    struct_name: &Ident,
) -> Result<(), syn::Error> {
    match &analyzed.base {
        AnalyzedBase::Struct(ty) if is_direct_self_type(ty, struct_name) => Err(
            diagnostics::direct_self_reference(ty, field_display_name, struct_name),
        ),
        AnalyzedBase::Tuple(elements) => {
            for element in elements {
                reject_direct_self_reference(element, field_display_name, struct_name)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
