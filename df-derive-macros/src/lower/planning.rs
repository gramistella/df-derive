use crate::ir::{
    FieldIR, FieldPlan, FieldSource, LeafSpec, NonEmpty, TerminalLeafSpec, TupleElement, TupleNode,
    TupleProjectionStep, column_name_for_ident,
};

pub fn plan_fields(fields: Vec<FieldIR>) -> Vec<FieldPlan> {
    fields.into_iter().map(plan_field).collect()
}

fn plan_field(field: FieldIR) -> FieldPlan {
    let source = FieldSource {
        name: field.name.clone(),
        field_index: field.field_index,
        outer_smart_ptr_depth: field.outer_smart_ptr_depth,
    };
    let name = column_name_for_ident(&field.name);
    match field.leaf_spec {
        LeafSpec::Tuple(elements) => FieldPlan::tuple(
            source,
            field.wrapper_shape,
            plan_tuple_elements(&name, elements),
        ),
        leaf_spec => FieldPlan::column(
            name,
            source,
            terminal_leaf(leaf_spec),
            field.wrapper_shape,
            field.nested_name_policy,
        ),
    }
}

fn plan_tuple_elements(column_prefix: &str, elements: Vec<TupleElement>) -> NonEmpty<TupleNode> {
    let nodes = elements
        .into_iter()
        .enumerate()
        .map(|(index, element)| plan_tuple_element(column_prefix, index, element))
        .collect();
    NonEmpty::from_vec(nodes).expect("type analysis rejects empty tuple fields")
}

fn plan_tuple_element(column_prefix: &str, index: usize, element: TupleElement) -> TupleNode {
    let name = format!("{column_prefix}.field_{index}");
    let step = TupleProjectionStep {
        index,
        outer_smart_ptr_depth: element.outer_smart_ptr_depth,
    };
    match element.leaf_spec {
        LeafSpec::Tuple(elements) => TupleNode::tuple(
            step,
            element.wrapper_shape,
            plan_tuple_elements(&name, elements),
        ),
        leaf_spec => TupleNode::leaf(name, step, element.wrapper_shape, terminal_leaf(leaf_spec)),
    }
}

fn terminal_leaf(leaf: LeafSpec) -> TerminalLeafSpec {
    TerminalLeafSpec::new(leaf).expect("tuple planning only emits terminal column leaves")
}
