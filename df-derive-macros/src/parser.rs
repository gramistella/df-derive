use crate::ir::{FieldIR, StructIR};
use crate::lower::field::lower_field;
use quote::format_ident;
use syn::{Data, DeriveInput, Fields, Ident};

fn validate_struct_input(input: &DeriveInput) -> Result<&syn::DataStruct, syn::Error> {
    match &input.data {
        Data::Struct(data_struct) => Ok(data_struct),
        Data::Enum(data_enum) => Err(syn::Error::new(
            data_enum.enum_token.span,
            "df-derive cannot be derived on enums; derive `ToDataFrame` on a struct \
             and use `#[df_derive(as_string)]` on enum fields",
        )),
        Data::Union(data_union) => Err(syn::Error::new(
            data_union.union_token.span,
            "df-derive cannot be derived on unions; derive `ToDataFrame` on a struct",
        )),
    }
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

    let data_struct = validate_struct_input(input)?;

    match &data_struct.fields {
        Fields::Named(named) => {
            for field in &named.named {
                let name_ident = field
                    .ident
                    .as_ref()
                    .expect("named fields must have ident")
                    .clone();
                if let Some(field_ir) =
                    lower_field(field, name_ident, None, &name, &generic_params)?
                {
                    fields_ir.push(field_ir);
                }
            }
        }
        Fields::Unit => {}
        Fields::Unnamed(unnamed) => {
            for (index, field) in unnamed.unnamed.iter().enumerate() {
                let name_ident = format_ident!("field_{}", index);
                if let Some(field_ir) =
                    lower_field(field, name_ident, Some(index), &name, &generic_params)?
                {
                    fields_ir.push(field_ir);
                }
            }
        }
    }

    Ok(StructIR {
        name,
        generics,
        fields: fields_ir,
    })
}
