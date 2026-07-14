use crate::attrs::{
    FieldConversion, FieldDisposition, FlattenConfig, LeafOverride, Spanned,
    parse_field_disposition,
};
use crate::ir::{FieldIR, NestedNamePolicy, WrapperShape};
use crate::lower::binary::parse_as_binary_shape;
use crate::lower::leaf::parse_leaf_spec;
use crate::lower::tuple::{FieldAttrRef, reject_attrs_on_tuple};
use crate::lower::validation::reject_direct_self_reference;
use crate::lower::wrappers::normalize_wrappers;
use crate::type_analysis::{AnalyzedBase, analyze_type};
use syn::Ident;

fn flatten_name_policy(config: &FlattenConfig) -> NestedNamePolicy {
    config
        .prefix
        .as_ref()
        .map_or(NestedNamePolicy::Flatten, |prefix| {
            NestedNamePolicy::Prefix(prefix.clone())
        })
}

fn reject_invalid_flatten_field(
    field: &syn::Field,
    field_display_name: &str,
    analyzed: &crate::type_analysis::AnalyzedType,
    wrapper_shape: &WrapperShape,
) -> Result<(), syn::Error> {
    let nested_base = matches!(
        analyzed.base,
        AnalyzedBase::Struct(_) | AnalyzedBase::Generic(_)
    );
    let bare_shape = matches!(wrapper_shape, WrapperShape::Leaf(shape) if shape.is_bare());
    if nested_base && bare_shape {
        return Ok(());
    }

    Err(syn::Error::new_spanned(
        field,
        format!(
            "field `{field_display_name}` has `flatten`, but flatten is only supported for \
             bare nested row fields after transparent pointer peeling. Use a concrete \
             struct or generic row payload that implements `Columnar`; \
             nullable, list, tuple, primitive, and conversion-shaped fields must remain \
             prefixed."
        ),
    ))
}

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

    let outer_smart_ptr_depth = analyzed.outer_smart_ptr_depth;
    let conversion = match &disposition {
        FieldDisposition::Include(conversion) => conversion,
        FieldDisposition::Skip => unreachable!("skip disposition returned before type analysis"),
    };
    let leaf_override: Option<&Spanned<LeafOverride>> = match conversion {
        FieldConversion::Default | FieldConversion::Binary { .. } | FieldConversion::Flatten(_) => {
            None
        }
        FieldConversion::LeafOverride(override_) => Some(override_),
    };
    let leaf_override_value = leaf_override.map(|override_| &override_.value);

    let normalized_wrappers = normalize_wrappers(&analyzed.wrappers);
    if let FieldConversion::Flatten(config) = conversion {
        reject_invalid_flatten_field(field, &display_name, &analyzed, &normalized_wrappers)?;
        return Ok(Some(FieldIR {
            name: name_ident,
            field_index,
            leaf_spec: parse_leaf_spec(field, &display_name, None, None, analyzed.base)?,
            wrapper_shape: normalized_wrappers,
            outer_smart_ptr_depth,
            nested_name_policy: flatten_name_policy(&config.value),
        }));
    }

    let (leaf_spec, wrapper_shape) = match conversion {
        FieldConversion::Binary { span } => {
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
        }
        FieldConversion::Default | FieldConversion::LeafOverride(_) => {
            let leaf_override_span = leaf_override.map(|override_| override_.span);
            let leaf = parse_leaf_spec(
                field,
                &display_name,
                leaf_override_value,
                leaf_override_span,
                analyzed.base,
            )?;
            (leaf, normalized_wrappers)
        }
        FieldConversion::Flatten(_) => unreachable!("flatten conversion returned above"),
    };

    Ok(Some(FieldIR {
        name: name_ident,
        field_index,
        leaf_spec,
        wrapper_shape,
        outer_smart_ptr_depth,
        nested_name_policy: NestedNamePolicy::Field,
    }))
}
