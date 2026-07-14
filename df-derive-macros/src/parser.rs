use crate::ir::{FieldIR, StructIR};
use crate::lower::{lower_field, plan_fields};
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
        fields: plan_fields(fields_ir),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        AccessStep, DecimalBackend, FieldColumn, FieldPlan, LeafSpec, NumericKind, TupleField,
        TupleNode, TupleNodeKind, WrapperShape,
    };

    fn parse(input: &DeriveInput) -> StructIR {
        parse_to_ir(input).expect("input should lower to IR")
    }

    fn field<'a>(ir: &'a StructIR, name: &str) -> &'a FieldPlan {
        ir.fields
            .iter()
            .find(|field| match field {
                FieldPlan::Column(column) => column.source().name == name,
                FieldPlan::Tuple(tuple) => tuple.source().name == name,
            })
            .expect("field plan should exist")
    }

    fn column<'a>(ir: &'a StructIR, name: &str) -> &'a FieldColumn {
        let FieldPlan::Column(column) = field(ir, name) else {
            panic!("expected a direct column field");
        };
        column
    }

    fn tuple<'a>(ir: &'a StructIR, name: &str) -> &'a TupleField {
        let FieldPlan::Tuple(tuple) = field(ir, name) else {
            panic!("expected a tuple field");
        };
        tuple
    }

    fn node<'a>(tuple: &'a TupleField, path: &[usize]) -> &'a TupleNode {
        let (first, rest) = path.split_first().expect("tuple path must not be empty");
        let mut node = &tuple.elements()[*first];
        for index in rest {
            let TupleNodeKind::Tuple(elements) = node.kind() else {
                panic!("tuple path traversed through a terminal node");
            };
            node = &elements[*index];
        }
        node
    }

    fn node_leaf_spec(node: &TupleNode) -> &crate::ir::TerminalLeafSpec {
        let TupleNodeKind::Leaf(common) = node.kind() else {
            panic!("expected a terminal tuple node");
        };
        common.leaf_spec()
    }

    fn assert_leaf_option_layers(shape: &WrapperShape, expected: usize) {
        let WrapperShape::Leaf(shape) = shape else {
            panic!("expected leaf wrapper shape");
        };
        assert_eq!(shape.access().option_layers(), expected);
    }

    fn assert_vec_shape(shape: &WrapperShape, outer_options: &[usize], inner_options: usize) {
        let WrapperShape::Vec(shape) = shape else {
            panic!("expected vec wrapper shape");
        };
        assert_eq!(shape.depth(), outer_options.len());
        for (idx, expected) in outer_options.iter().copied().enumerate() {
            assert_eq!(shape.layers[idx].access.option_layers(), expected);
        }
        assert_eq!(shape.inner_access.option_layers(), inner_options);
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
                nested_optional_vec_tuple: Vec<Option<Option<(Vec<i32>, Option<String>)>>>,
            }
        });

        let doubly_optional = column(&ir, "doubly_optional");
        assert_leaf_option_layers(doubly_optional.wrapper_shape(), 2);
        assert!(matches!(
            doubly_optional.leaf_spec().as_leaf_spec(),
            LeafSpec::Generic(ident) if ident == "T"
        ));

        assert_vec_shape(column(&ir, "optional_vec").wrapper_shape(), &[1], 0);
        assert_vec_shape(column(&ir, "vec_optional").wrapper_shape(), &[0], 1);
        assert_vec_shape(column(&ir, "vec_option_vec").wrapper_shape(), &[0, 1], 0);
        assert_vec_shape(column(&ir, "option_vec_option").wrapper_shape(), &[1], 1);

        let optional_tuple = tuple(&ir, "optional_tuple");
        assert_leaf_option_layers(optional_tuple.wrapper_shape(), 1);
        let optional_tuple_0 = node(optional_tuple, &[0]);
        assert_leaf_option_layers(optional_tuple_0.wrapper_shape(), 0);
        assert!(matches!(
            node_leaf_spec(optional_tuple_0).as_leaf_spec(),
            LeafSpec::Numeric(NumericKind::I32)
        ));
        assert!(matches!(
            node_leaf_spec(node(optional_tuple, &[1])).as_leaf_spec(),
            LeafSpec::String
        ));

        let vec_tuple = tuple(&ir, "vec_tuple");
        assert_vec_shape(vec_tuple.wrapper_shape(), &[0], 0);
        assert_vec_shape(node(vec_tuple, &[0]).wrapper_shape(), &[0], 0);
        assert_leaf_option_layers(node(vec_tuple, &[1]).wrapper_shape(), 1);

        let nested_tuple = tuple(&ir, "nested_optional_vec_tuple");
        assert_vec_shape(nested_tuple.wrapper_shape(), &[0], 2);
        let WrapperShape::Vec(nested_parent) = nested_tuple.wrapper_shape() else {
            unreachable!("assert_vec_shape already proved this is a Vec shape");
        };
        assert_eq!(
            nested_parent.inner_access.iter().collect::<Vec<_>>(),
            [AccessStep::Option, AccessStep::Option]
        );
        assert_vec_shape(node(nested_tuple, &[0]).wrapper_shape(), &[0], 0);
        assert_leaf_option_layers(node(nested_tuple, &[1]).wrapper_shape(), 1);
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
        let FieldPlan::Column(kept) = &ir.fields[0] else {
            panic!("kept field should be a direct column");
        };
        assert_eq!(kept.name(), "kept");
        assert!(matches!(
            kept.wrapper_shape(),
            WrapperShape::Leaf(shape) if shape.is_bare()
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
            column(&ir, "runtime").leaf_spec().as_leaf_spec(),
            LeafSpec::Decimal {
                precision: 38,
                scale: 10,
                backend: DecimalBackend::RuntimeKnown,
            }
        ));

        assert!(matches!(
            column(&ir, "generic").leaf_spec().as_leaf_spec(),
            LeafSpec::Decimal {
                precision: 12,
                scale: 3,
                backend: DecimalBackend::Generic(ident),
            } if ident == "T"
        ));

        assert!(matches!(
            column(&ir, "custom").leaf_spec().as_leaf_spec(),
            LeafSpec::Decimal {
                precision: 18,
                scale: 4,
                backend: DecimalBackend::Struct(_),
            }
        ));
    }

    #[test]
    fn retains_tuple_hierarchy_and_terminal_column_order() {
        let ir = parse(&syn::parse_quote! {
            struct Row {
                bare: (i32, String),
                optional: Option<(Vec<i32>, String)>,
                vec_parent: Vec<(Vec<i32>, Option<String>)>,
                boxed_nested: Box<((i32, String), std::sync::Arc<bool>)>,
            }
        });

        let mut names = Vec::new();
        ir.visit_terminal_columns(|column| names.push(column.name().to_owned()));
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

        let optional = tuple(&ir, "optional");
        assert_leaf_option_layers(optional.wrapper_shape(), 1);
        assert_vec_shape(node(optional, &[0]).wrapper_shape(), &[0], 0);
        assert_leaf_option_layers(node(optional, &[1]).wrapper_shape(), 0);

        let vec_parent = tuple(&ir, "vec_parent");
        assert_vec_shape(vec_parent.wrapper_shape(), &[0], 0);
        assert_vec_shape(node(vec_parent, &[0]).wrapper_shape(), &[0], 0);
        assert_leaf_option_layers(node(vec_parent, &[1]).wrapper_shape(), 1);

        let boxed_nested = tuple(&ir, "boxed_nested");
        assert_eq!(boxed_nested.source().outer_smart_ptr_depth, 1);
        let inner_tuple = node(boxed_nested, &[0]);
        assert!(matches!(inner_tuple.kind(), TupleNodeKind::Tuple(_)));
        assert_eq!(inner_tuple.step().index, 0);
        assert_eq!(node(boxed_nested, &[1]).step().outer_smart_ptr_depth, 1);
        assert!(matches!(
            node_leaf_spec(node(boxed_nested, &[1])).as_leaf_spec(),
            LeafSpec::Bool
        ));
    }

    #[test]
    fn accepts_wrappers_around_nested_tuple_nodes() {
        let ir = parse(&syn::parse_quote! {
            struct Row {
                optional_parent: Option<((i32, String), bool)>,
                optional_element: (Option<(i32, String)>, bool),
                list_parent: Vec<((i32, String), bool)>,
            }
        });

        let optional_parent = tuple(&ir, "optional_parent");
        assert_leaf_option_layers(optional_parent.wrapper_shape(), 1);
        assert_leaf_option_layers(node(optional_parent, &[0]).wrapper_shape(), 0);

        let optional_element = tuple(&ir, "optional_element");
        assert_leaf_option_layers(optional_element.wrapper_shape(), 0);
        assert_leaf_option_layers(node(optional_element, &[0]).wrapper_shape(), 1);

        let list_parent = tuple(&ir, "list_parent");
        assert_vec_shape(list_parent.wrapper_shape(), &[0], 0);
        let nested = node(list_parent, &[0]);
        assert!(matches!(nested.kind(), TupleNodeKind::Tuple(_)));
        assert_leaf_option_layers(nested.wrapper_shape(), 0);

        let mut list_columns = Vec::new();
        ir.visit_terminal_columns(|column| {
            if column.name().starts_with("list_parent.") {
                list_columns.push((column.name().to_owned(), column.vec_depth()));
            }
        });
        assert_eq!(
            list_columns,
            [
                ("list_parent.field_0.field_0".to_owned(), 1),
                ("list_parent.field_0.field_1".to_owned(), 1),
                ("list_parent.field_1".to_owned(), 1),
            ]
        );
    }
}
