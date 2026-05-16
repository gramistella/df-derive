use crate::ir::{FieldIR, StructIR};
use crate::lower::lower_field;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DecimalBackend, LeafShape, LeafSpec, NumericKind, WrapperShape};

    fn parse(input: &DeriveInput) -> StructIR {
        parse_to_ir(input).expect("input should lower to IR")
    }

    fn field<'a>(ir: &'a StructIR, name: &str) -> &'a FieldIR {
        ir.fields
            .iter()
            .find(|field| field.name == name)
            .expect("field should exist")
    }

    fn assert_leaf_option_layers(shape: &WrapperShape, expected: usize) {
        let WrapperShape::Leaf(shape) = shape else {
            panic!("expected leaf wrapper shape");
        };
        assert_eq!(shape.option_layers(), expected);
    }

    fn assert_vec_shape(shape: &WrapperShape, outer_options: &[usize], inner_options: usize) {
        let WrapperShape::Vec(shape) = shape else {
            panic!("expected vec wrapper shape");
        };
        assert_eq!(shape.depth(), outer_options.len());
        for (idx, expected) in outer_options.iter().copied().enumerate() {
            assert_eq!(shape.layers[idx].option_layers_above, expected);
        }
        assert_eq!(shape.inner_option_layers, inner_options);
    }

    #[test]
    fn lowers_option_vec_and_tuple_wrapper_shapes() {
        let ir = parse(&syn::parse_quote! {
            struct Row<T> {
                doubly_optional: Option<Option<T>>,
                optional_vec: Option<Vec<T>>,
                vec_optional: Vec<Option<T>>,
                vec_option_vec: Vec<Option<Vec<T>>>,
                option_vec_option: Option<Vec<Option<T>>>,
                optional_tuple: Option<(i32, String)>,
                vec_tuple: Vec<(Vec<i32>, Option<String>)>,
            }
        });

        let doubly_optional = field(&ir, "doubly_optional");
        assert_leaf_option_layers(&doubly_optional.wrapper_shape, 2);
        assert!(matches!(
            doubly_optional.leaf_spec,
            LeafSpec::Generic(ref ident) if ident == "T"
        ));

        assert_vec_shape(&field(&ir, "optional_vec").wrapper_shape, &[1], 0);
        assert_vec_shape(&field(&ir, "vec_optional").wrapper_shape, &[0], 1);
        assert_vec_shape(&field(&ir, "vec_option_vec").wrapper_shape, &[0, 1], 0);
        assert_vec_shape(&field(&ir, "option_vec_option").wrapper_shape, &[1], 1);

        let optional_tuple = field(&ir, "optional_tuple");
        assert_leaf_option_layers(&optional_tuple.wrapper_shape, 1);
        let LeafSpec::Tuple(elements) = &optional_tuple.leaf_spec else {
            panic!("expected tuple leaf");
        };
        assert_eq!(elements.len(), 2);
        assert!(matches!(
            elements[0].leaf_spec,
            LeafSpec::Numeric(NumericKind::I32)
        ));
        assert!(matches!(elements[1].leaf_spec, LeafSpec::String));

        let vec_tuple = field(&ir, "vec_tuple");
        assert_vec_shape(&vec_tuple.wrapper_shape, &[0], 0);
        let LeafSpec::Tuple(elements) = &vec_tuple.leaf_spec else {
            panic!("expected tuple leaf");
        };
        assert_eq!(elements.len(), 2);
        assert_vec_shape(&elements[0].wrapper_shape, &[0], 0);
        assert_leaf_option_layers(&elements[1].wrapper_shape, 1);
    }

    #[test]
    fn skip_attribute_omits_field_from_ir() {
        let ir = parse(&syn::parse_quote! {
            struct Row {
                kept: u32,
                #[df_derive(skip)]
                skipped: String,
            }
        });

        assert_eq!(ir.fields.len(), 1);
        assert_eq!(ir.fields[0].name, "kept");
        assert!(matches!(
            ir.fields[0].wrapper_shape,
            WrapperShape::Leaf(LeafShape::Bare)
        ));
    }

    #[test]
    fn decimal_backend_is_part_of_leaf_spec() {
        let ir = parse(&syn::parse_quote! {
            struct Row<T> {
                runtime: Decimal,
                #[df_derive(decimal(precision = 12, scale = 3))]
                generic: T,
                #[df_derive(decimal(precision = 18, scale = 4))]
                custom: CustomDecimal,
            }
        });

        assert!(matches!(
            field(&ir, "runtime").leaf_spec,
            LeafSpec::Decimal {
                precision: 38,
                scale: 10,
                backend: DecimalBackend::RuntimeKnown,
            }
        ));

        assert!(matches!(
            &field(&ir, "generic").leaf_spec,
            LeafSpec::Decimal {
                precision: 12,
                scale: 3,
                backend: DecimalBackend::Generic(ident),
            } if ident == "T"
        ));

        assert!(matches!(
            &field(&ir, "custom").leaf_spec,
            LeafSpec::Decimal {
                precision: 18,
                scale: 4,
                backend: DecimalBackend::Struct(_),
            }
        ));
    }
}
