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

    match segments.len() {
        1 => {
            let Some(last) = segments.last() else {
                return false;
            };

            last.ident == "Self" || last.ident == *struct_name
        }
        2 => {
            let Some(first) = segments.first() else {
                return false;
            };
            let Some(last) = segments.last() else {
                return false;
            };

            matches!(first.arguments, syn::PathArguments::None)
                && (first.ident == "crate" || first.ident == "self")
                && last.ident == *struct_name
        }
        _ => false,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_direct(ty: &syn::Type) {
        assert!(is_direct_self_type(ty, &syn::parse_quote!(Node)));
    }

    #[test]
    fn rejects_direct_self_recursion_shapes() {
        assert_direct(&syn::parse_quote!(Self));
        assert_direct(&syn::parse_quote!(Node<T>));
        assert_direct(&syn::parse_quote!(crate::Node<T>));
        assert_direct(&syn::parse_quote!(self::Node<T>));
    }
}
