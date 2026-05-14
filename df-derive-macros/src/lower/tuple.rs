use crate::attrs::field::{FieldOverride, LeafOverride};
use crate::diagnostics;
use crate::ir::TupleElement;
use crate::lower::field::default_leaf_for_base;
use crate::lower::wrappers::normalize_wrappers;
use crate::type_analysis::{AnalyzedBase, AnalyzedType, RawWrapper};

#[derive(Clone, Copy)]
pub enum FieldOverrideRef<'a> {
    Field(&'a FieldOverride),
    Leaf(&'a LeafOverride),
}

/// Reject every field-level override on a tuple-typed field with a
/// per-attribute message. Attributes apply to a single column's leaf
/// classification and have no per-element selector, so `as_str` / `as_string`
/// / `as_binary` / `decimal(...)` / `time_unit = "..."` over a tuple is
/// always ambiguous. The fix is to hoist the tuple into a named struct
/// where per-element attributes can be applied at field level.
pub fn reject_attrs_on_tuple(
    field: &syn::Field,
    field_display_name: &str,
    override_: Option<FieldOverrideRef<'_>>,
) -> Result<(), syn::Error> {
    let attr = match override_ {
        None | Some(FieldOverrideRef::Field(FieldOverride::Skip)) => return Ok(()),
        Some(FieldOverrideRef::Field(FieldOverride::AsBinary)) => "as_binary",
        Some(
            FieldOverrideRef::Field(FieldOverride::Leaf(override_))
            | FieldOverrideRef::Leaf(override_),
        ) => match override_ {
            LeafOverride::AsStr => "as_str",
            LeafOverride::AsString => "as_string",
            LeafOverride::Decimal { .. } => "decimal(...)",
            LeafOverride::TimeUnit(_) => "time_unit = \"...\"",
        },
    };
    Err(diagnostics::unsupported_tuple_attr(
        field,
        field_display_name,
        attr,
    ))
}

/// Lower one analyzed tuple element to its IR form. Recurses into nested
/// tuples (`((i32, String), bool)`), preserves outer smart-pointer counts on
/// the element, and normalizes the element's wrapper stack independently of
/// the parent's. Field-level attributes are not applied here — they are
/// rejected on the parent field by [`reject_attrs_on_tuple`] before this runs.
pub fn analyzed_to_tuple_element(
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

pub fn reject_unsupported_wrapped_nested_tuples(
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
