use crate::ir::{FieldIR, StructIR};
use crate::lower::{lower_field, project_fields_to_columns};
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
        columns: project_fields_to_columns(fields_ir),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ColumnIR, ColumnSource, DecimalBackend, LeafShape, LeafSpec, NumericKind,
        ProjectionContext, WrapperShape,
    };

    fn parse(input: &DeriveInput) -> StructIR {
        parse_to_ir(input).expect("input should lower to IR")
    }

    fn column<'a>(ir: &'a StructIR, name: &str) -> &'a ColumnIR {
        ir.columns
            .iter()
            .find(|column| column.name == name)
            .expect("column should exist")
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

        let doubly_optional = column(&ir, "doubly_optional");
        assert_leaf_option_layers(&doubly_optional.wrapper_shape, 2);
        assert!(matches!(
            doubly_optional.leaf_spec.as_leaf_spec(),
            LeafSpec::Generic(ident) if ident == "T"
        ));

        assert_vec_shape(&column(&ir, "optional_vec").wrapper_shape, &[1], 0);
        assert_vec_shape(&column(&ir, "vec_optional").wrapper_shape, &[0], 1);
        assert_vec_shape(&column(&ir, "vec_option_vec").wrapper_shape, &[0, 1], 0);
        assert_vec_shape(&column(&ir, "option_vec_option").wrapper_shape, &[1], 1);

        let optional_tuple_0 = column(&ir, "optional_tuple.field_0");
        assert_leaf_option_layers(&optional_tuple_0.wrapper_shape, 1);
        assert!(matches!(
            optional_tuple_0.leaf_spec.as_leaf_spec(),
            LeafSpec::Numeric(NumericKind::I32)
        ));
        assert!(matches!(
            optional_tuple_0.source,
            ColumnSource::TupleProjection {
                context: ProjectionContext::ParentOption { .. },
                ..
            }
        ));
        assert!(matches!(
            column(&ir, "optional_tuple.field_1")
                .leaf_spec
                .as_leaf_spec(),
            LeafSpec::String
        ));

        assert_vec_shape(&column(&ir, "vec_tuple.field_0").wrapper_shape, &[0, 0], 0);
        assert_vec_shape(&column(&ir, "vec_tuple.field_1").wrapper_shape, &[0], 1);
        assert!(matches!(
            column(&ir, "vec_tuple.field_0").source,
            ColumnSource::TupleProjection {
                context: ProjectionContext::ParentVec { .. },
                ..
            }
        ));
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

        assert_eq!(ir.columns.len(), 1);
        assert_eq!(ir.columns[0].name, "kept");
        assert!(matches!(
            ir.columns[0].wrapper_shape,
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
            column(&ir, "runtime").leaf_spec.as_leaf_spec(),
            LeafSpec::Decimal {
                precision: 38,
                scale: 10,
                backend: DecimalBackend::RuntimeKnown,
            }
        ));

        assert!(matches!(
            column(&ir, "generic").leaf_spec.as_leaf_spec(),
            LeafSpec::Decimal {
                precision: 12,
                scale: 3,
                backend: DecimalBackend::Generic(ident),
            } if ident == "T"
        ));

        assert!(matches!(
            column(&ir, "custom").leaf_spec.as_leaf_spec(),
            LeafSpec::Decimal {
                precision: 18,
                scale: 4,
                backend: DecimalBackend::Struct(_),
            }
        ));
    }

    #[test]
    fn projects_tuple_fields_to_terminal_columns() {
        let ir = parse(&syn::parse_quote! {
            struct Row {
                bare: (i32, String),
                optional: Option<(Vec<i32>, String)>,
                vec_parent: Vec<(Vec<i32>, Option<String>)>,
                boxed_nested: Box<((i32, String), std::sync::Arc<bool>)>,
            }
        });

        let names: Vec<&str> = ir
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "bare.field_0",
                "bare.field_1",
                "optional.field_0",
                "optional.field_1",
                "vec_parent.field_0",
                "vec_parent.field_1",
                "boxed_nested.field_0.field_0",
                "boxed_nested.field_0.field_1",
                "boxed_nested.field_1",
            ]
        );

        assert!(matches!(
            column(&ir, "bare.field_0").source,
            ColumnSource::TupleProjection {
                context: ProjectionContext::Static,
                ..
            }
        ));
        assert_vec_shape(&column(&ir, "optional.field_0").wrapper_shape, &[1], 0);
        assert_leaf_option_layers(&column(&ir, "optional.field_1").wrapper_shape, 1);
        assert_vec_shape(&column(&ir, "vec_parent.field_0").wrapper_shape, &[0, 0], 0);
        assert_vec_shape(&column(&ir, "vec_parent.field_1").wrapper_shape, &[0], 1);
        assert!(matches!(
            column(&ir, "boxed_nested.field_1").leaf_spec.as_leaf_spec(),
            LeafSpec::Bool
        ));
    }
}
