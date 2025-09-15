use crate::ir::{FieldIR, PrimitiveTransform, StructIR};
use crate::type_analysis::analyze_type;
use quote::format_ident;
use syn::{Data, DeriveInput, Fields};

#[derive(Default, Clone, Copy)]
struct FieldAttributes {
    as_string: bool,
}

fn parse_attributes(field: &syn::Field) -> FieldAttributes {
    let mut attrs = FieldAttributes::default();
    for attr in &field.attrs {
        if attr.path().is_ident("df_derive") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("as_string") {
                    attrs.as_string = true;
                }
                Ok(())
            });
        }
    }
    attrs
}

/// Parse a `syn::DeriveInput` into the new IR used by the next-gen codegen.
///
/// This function is intentionally infallible for unsupported inputs (e.g., enums
/// or tuple structs) to avoid breaking existing error reporting flow. In such
/// cases it produces an empty `StructIR` that is currently unused by codegen.
pub fn parse_to_ir(input: &DeriveInput) -> Result<StructIR, syn::Error> {
    let name = input.ident.clone();
    let mut fields_ir: Vec<FieldIR> = Vec::new();

    if let Data::Struct(data_struct) = &input.data {
        match &data_struct.fields {
            Fields::Named(named) => {
                for field in &named.named {
                    let field_name_ident = field
                        .ident
                        .as_ref()
                        .expect("named fields must have ident")
                        .clone();

                    let attrs = parse_attributes(field);
                    let analyzed = analyze_type(&field.ty, attrs.as_string)?;

                    let base_type = analyzed.base.clone();
                    let transform = analyzed.transform.clone();
                    // Note: as_string attribute is represented as transform=ToString; codegen will interpret
                    let _ = PrimitiveTransform::ToString; // keep enum referenced for linking
                    fields_ir.push(FieldIR {
                        name: field_name_ident,
                        field_index: None,
                        wrappers: analyzed.wrappers.clone(),
                        base_type,
                        transform,
                    });
                }
            }
            Fields::Unit => {
                // Unit structs have no fields, so no fields IR needed.
            }
            Fields::Unnamed(unnamed) => {
                // Handle tuple structs by creating field IR for each unnamed field
                for (index, field) in unnamed.unnamed.iter().enumerate() {
                    let field_name_ident = format_ident!("field_{}", index);

                    let attrs = parse_attributes(field);
                    let analyzed = analyze_type(&field.ty, attrs.as_string)?;

                    let base_type = analyzed.base.clone();
                    let transform = analyzed.transform.clone();
                    fields_ir.push(FieldIR {
                        name: field_name_ident,
                        field_index: Some(index),
                        wrappers: analyzed.wrappers.clone(),
                        base_type,
                        transform,
                    });
                }
            }
        }
    } else {
        // Non-struct inputs produce an empty IR for now.
    }

    Ok(StructIR {
        name,
        fields: fields_ir,
    })
}

// Old field-based flattening and custom-struct codegen helpers removed
