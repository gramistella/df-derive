use crate::attrs::{FieldDisposition, LeafOverride, Spanned, parse_field_disposition};
use crate::ir::FieldIR;
use crate::lower::binary::parse_as_binary_shape;
use crate::lower::decimal::{decimal_backend_ty_for_override, decimal_generic_params_for_override};
use crate::lower::leaf::parse_leaf_spec;
use crate::lower::tuple::{
    FieldAttrRef, reject_attrs_on_tuple, reject_unsupported_wrapped_nested_tuples,
};
use crate::lower::validation::reject_direct_self_reference;
use crate::lower::wrappers::normalize_wrappers;
use crate::type_analysis::{AnalyzedBase, analyze_type};
use syn::Ident;

pub fn lower_field(
    field: &syn::Field,
    name_ident: Ident,
    field_index: Option<usize>,
    struct_name: &Ident,
    generic_params: &[Ident],
) -> Result<Option<FieldIR>, syn::Error> {
    let display_name = name_ident.to_string();
    let disposition = parse_field_disposition(field, &display_name)?;
    if matches!(disposition, FieldDisposition::Skip) {
        return Ok(None);
    }

    let analyzed = analyze_type(&field.ty, generic_params)?;
    reject_direct_self_reference(&analyzed, &display_name, struct_name)?;
    reject_unsupported_wrapped_nested_tuples(&analyzed, &display_name)?;

    let outer_smart_ptr_depth = analyzed.outer_smart_ptr_depth;
    let leaf_override: Option<&Spanned<LeafOverride>> = match &disposition {
        FieldDisposition::Include { leaf_override } => leaf_override.as_ref(),
        FieldDisposition::Skip => unreachable!("skip disposition returned before type analysis"),
        FieldDisposition::Binary { .. } => None,
    };
    let leaf_override_value = leaf_override.map(|override_| &override_.value);
    let decimal_generic_params =
        decimal_generic_params_for_override(leaf_override_value, &analyzed.base);
    let decimal_backend_ty = decimal_backend_ty_for_override(leaf_override_value, &analyzed.base);

    let (leaf_spec, wrapper_shape) = if let FieldDisposition::Binary { span } = &disposition {
        if matches!(analyzed.base, AnalyzedBase::Tuple(_)) {
            reject_attrs_on_tuple(
                field,
                &display_name,
                Some(FieldAttrRef::Binary { span: *span }),
            )?;
        }
        let (leaf, trimmed) =
            parse_as_binary_shape(field, &display_name, &analyzed.base, &analyzed.wrappers)?;
        (leaf, normalize_wrappers(&trimmed))
    } else {
        let leaf_override_span = leaf_override.map(|override_| override_.span);
        let leaf = parse_leaf_spec(
            field,
            &display_name,
            leaf_override_value,
            leaf_override_span,
            analyzed.base,
        )?;
        (leaf, normalize_wrappers(&analyzed.wrappers))
    };

    Ok(Some(FieldIR {
        name: name_ident,
        field_index,
        leaf_spec,
        wrapper_shape,
        decimal_generic_params,
        decimal_backend_ty,
        outer_smart_ptr_depth,
    }))
}
