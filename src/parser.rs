use crate::ir::{FieldIR, StructIR};
use crate::type_analysis::analyze_type;
use quote::format_ident;
use syn::{Data, DeriveInput, Fields, Ident};

#[derive(Default, Clone, Copy)]
struct FieldAttributes {
    as_string: bool,
    as_str: bool,
}

fn parse_attributes(
    field: &syn::Field,
    field_display_name: &str,
) -> Result<FieldAttributes, syn::Error> {
    let mut attrs = FieldAttributes::default();
    for attr in &field.attrs {
        if attr.path().is_ident("df_derive") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("as_string") {
                    attrs.as_string = true;
                    Ok(())
                } else if meta.path.is_ident("as_str") {
                    attrs.as_str = true;
                    Ok(())
                } else {
                    // Reject unknown keys so a typo (e.g. `as_strg` or `as_string` swapped
                    // with `as_str`) surfaces at the user's source instead of being
                    // silently ignored.
                    Err(meta.error(
                        "unknown key in #[df_derive(...)] field attribute; expected `as_str` or `as_string`",
                    ))
                }
            })?;
        }
    }
    if attrs.as_string && attrs.as_str {
        return Err(syn::Error::new_spanned(
            field,
            format!(
                "field `{field_display_name}` has both `as_str` and `as_string`; \
                 pick one — `as_str` borrows via `AsRef<str>` (no allocation), \
                 `as_string` formats via `Display` (allocates per row)"
            ),
        ));
    }
    Ok(attrs)
}

/// Parse a `syn::DeriveInput` into the IR consumed by codegen.
///
/// Returns a `syn::Error` for non-struct inputs (enums, unions). Tuple structs
/// and unit structs are supported.
pub fn parse_to_ir(input: &DeriveInput) -> Result<StructIR, syn::Error> {
    let name = input.ident.clone();
    let generics = input.generics.clone();
    let generic_params: Vec<Ident> = generics.type_params().map(|tp| tp.ident.clone()).collect();
    let mut fields_ir: Vec<FieldIR> = Vec::new();

    let data_struct = match &input.data {
        Data::Struct(data_struct) => data_struct,
        Data::Enum(data_enum) => {
            return Err(syn::Error::new(
                data_enum.enum_token.span,
                "df-derive cannot be derived on enums; derive `ToDataFrame` on a struct \
                 and use `#[df_derive(as_string)]` on enum fields",
            ));
        }
        Data::Union(data_union) => {
            return Err(syn::Error::new(
                data_union.union_token.span,
                "df-derive cannot be derived on unions; derive `ToDataFrame` on a struct",
            ));
        }
    };

    match &data_struct.fields {
        Fields::Named(named) => {
            for field in &named.named {
                let field_name_ident = field
                    .ident
                    .as_ref()
                    .expect("named fields must have ident")
                    .clone();

                let display_name = field_name_ident.to_string();
                let attrs = parse_attributes(field, &display_name)?;
                let analyzed =
                    analyze_type(&field.ty, attrs.as_string, attrs.as_str, &generic_params)?;

                let base_type = analyzed.base.clone();
                let transform = analyzed.transform.clone();
                fields_ir.push(FieldIR {
                    name: field_name_ident,
                    field_index: None,
                    wrappers: analyzed.wrappers.clone(),
                    base_type,
                    transform,
                    field_ty: field.ty.clone(),
                });
            }
        }
        Fields::Unit => {}
        Fields::Unnamed(unnamed) => {
            for (index, field) in unnamed.unnamed.iter().enumerate() {
                let field_name_ident = format_ident!("field_{}", index);

                let display_name = field_name_ident.to_string();
                let attrs = parse_attributes(field, &display_name)?;
                let analyzed =
                    analyze_type(&field.ty, attrs.as_string, attrs.as_str, &generic_params)?;

                let base_type = analyzed.base.clone();
                let transform = analyzed.transform.clone();
                fields_ir.push(FieldIR {
                    name: field_name_ident,
                    field_index: Some(index),
                    wrappers: analyzed.wrappers.clone(),
                    base_type,
                    transform,
                    field_ty: field.ty.clone(),
                });
            }
        }
    }

    Ok(StructIR {
        name,
        generics,
        fields: fields_ir,
    })
}
