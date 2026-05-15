use crate::attrs::{FieldOverride, LeafOverride};
use crate::ir::TupleElement;
use crate::type_analysis::{AnalyzedBase, AnalyzedType, RawWrapper};
use proc_macro2::Span;

use super::diagnostics;
use super::field::default_leaf_for_base;
use super::wrappers::normalize_wrappers;

#[derive(Clone, Copy)]
pub(super) enum FieldOverrideRef<'a> {
    Field {
        value: &'a FieldOverride,
        span: Span,
    },
    Leaf {
        value: &'a LeafOverride,
        span: Span,
    },
}

/// Reject every field-level override on a tuple-typed field with a
/// per-attribute message. Attributes apply to a single column's leaf
/// classification and have no per-element selector, so `as_str` / `as_string`
/// / `as_binary` / `decimal(...)` / `time_unit = "..."` over a tuple is
/// always ambiguous. The fix is to hoist the tuple into a named struct
/// where per-element attributes can be applied at field level.
pub(super) fn reject_attrs_on_tuple(
    _field: &syn::Field,
    field_display_name: &str,
    override_: Option<FieldOverrideRef<'_>>,
) -> Result<(), syn::Error> {
    let (attr, span) = match override_ {
        None
        | Some(FieldOverrideRef::Field {
            value: FieldOverride::Skip,
            ..
        }) => return Ok(()),
        Some(FieldOverrideRef::Field {
            value: FieldOverride::AsBinary,
            span,
        }) => ("as_binary", span),
        Some(
            FieldOverrideRef::Field {
                value: FieldOverride::Leaf(override_),
                span,
            }
            | FieldOverrideRef::Leaf {
                value: override_,
                span,
            },
        ) => match override_ {
            LeafOverride::AsStr => ("as_str", span),
            LeafOverride::AsString => ("as_string", span),
            LeafOverride::Decimal { .. } => ("decimal(...)", span),
            LeafOverride::TimeUnit(_) => ("time_unit = \"...\"", span),
        },
    };
    Err(diagnostics::unsupported_tuple_attr_at(
        span,
        field_display_name,
        attr,
    ))
}

/// Lower one analyzed tuple element to its IR form. Recurses into nested
/// tuples (`((i32, String), bool)`), preserves outer smart-pointer counts on
/// the element, and normalizes the element's wrapper stack independently of
/// the parent's. Field-level attributes are not applied here — they are
/// rejected on the parent field by [`reject_attrs_on_tuple`] before this runs.
pub(super) fn analyzed_to_tuple_element(
    analyzed: AnalyzedType,
    field_display_name: &str,
) -> Result<TupleElement, syn::Error> {
    let leaf_spec =
        default_leaf_for_base(&analyzed.field_ty, field_display_name, analyzed.base, false)?;
    let wrapper_shape = normalize_wrappers(&analyzed.wrappers);
    Ok(TupleElement {
        leaf_spec,
        wrapper_shape,
        outer_smart_ptr_depth: analyzed.outer_smart_ptr_depth,
    })
}

const fn has_semantic_wrappers(wrappers: &[RawWrapper]) -> bool {
    !wrappers.is_empty()
}

pub(super) fn reject_unsupported_wrapped_nested_tuples(
    analyzed: &AnalyzedType,
    field_display_name: &str,
) -> Result<(), syn::Error> {
    let AnalyzedBase::Tuple(elements) = &analyzed.base else {
        return Ok(());
    };
    let parent_wrapped = has_semantic_wrappers(&analyzed.wrappers);

    for element in elements {
        if matches!(element.base, AnalyzedBase::Tuple(_))
            && (parent_wrapped || has_semantic_wrappers(&element.wrappers))
        {
            return Err(diagnostics::unsupported_wrapped_nested_tuple(
                &element.field_ty,
                field_display_name,
            ));
        }

        reject_unsupported_wrapped_nested_tuples(element, field_display_name)?;
    }

    Ok(())
}
